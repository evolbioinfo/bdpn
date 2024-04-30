import os
from multiprocessing.pool import ThreadPool

import numpy as np
import pandas as pd

from bdpn import bd_model
from bdpn.formulas import get_log_p, get_c1, get_c2, get_E2, get_E1, get_log_po, get_log_pn, get_log_po_from_p_pn, \
    get_u, get_log_pp
from bdpn.parameter_estimator import optimize_likelihood_params, rescale_log
from bdpn.tree_manager import TIME, read_forest, annotate_forest_with_time

PARAMETER_NAMES = np.array(['la', 'psi', 'partner_psi', 'p', 'pn'])

DEFAULT_LOWER_BOUNDS = [bd_model.DEFAULT_MIN_RATE, bd_model.DEFAULT_MIN_RATE, bd_model.DEFAULT_MIN_RATE,
                        bd_model.DEFAULT_MIN_PROB, bd_model.DEFAULT_MIN_PROB]
DEFAULT_UPPER_BOUNDS = [bd_model.DEFAULT_MAX_RATE, bd_model.DEFAULT_MAX_RATE, bd_model.DEFAULT_MAX_RATE * 1e3,
                        bd_model.DEFAULT_MAX_PROB, bd_model.DEFAULT_MAX_PROB]


def log_sum(log_summands):
    """
    Takes [logX1, ..., logXk] as input and returns log(X1 + ... + Xk) as output, 
    while taking care of potential under/overflow.
    
    :param log_summands: an array of summands in log form
    :return: log of the sum
    """
    result = np.array(log_summands, dtype=np.float64)
    factors = rescale_log(result)
    return np.log(np.sum(np.exp(result))) - factors


def loglikelihood(forest, la, psi, phi, rho, upsilon, T=None, threads=1):
    annotate_forest_with_time(forest)
    T = get_T(T, forest)

    c1 = get_c1(la=la, psi=psi, rho=rho)
    c2 = get_c2(la=la, psi=psi, c1=c1)

    log_la, log_psi, log_phi, log_rho, log_ups, log_not_ups, log_2 = \
        np.log(la), np.log(psi), np.log(phi), np.log(rho), np.log(upsilon), np.log(1 - upsilon), np.log(2)

    def process_node(node):
        """
        Calculate the loglikelihood density of this branch being unnotified, including the subtree
        and put it into node's annotation: 'lx' for internal nodes,
        'lxx' and 'lxn' (depending on whether the node notified) for tips.

        :param node: tree node whose children (if any) are already processed
        :return: void, add a node annotation
        """
        ti = getattr(node, TIME)
        tj = ti - node.dist

        E1 = get_E1(c1=c1, c2=c2, t=tj, T=T)
        E2 = get_E2(c1=c1, c2=c2, ti=ti, T=T)

        log_p = get_log_p(c1=c1, t=tj, ti=ti, E1=E1, E2=E2)
        log_pn = get_log_pn(la=la, psi=psi, t=tj, ti=ti)
        log_po = get_log_po_from_p_pn(log_p=log_p, log_pn=log_pn)

        if node.is_leaf():
            node.add_feature('lxx', log_p + log_psi + log_rho + log_not_ups)
            node.add_feature('lxn', log_pn + log_psi + log_rho + log_ups)
            node.add_feature('lnx_late', log_po + log_psi + log_rho + log_not_ups)
            node.add_feature('lnn_late', log_pn + log_psi + log_rho + log_ups)
            return

        node.add_feature('logp', log_p)
        node.add_feature('logpo', log_po)

        i0, i1 = node.children
        is_tip0, is_tip1 = i0.is_leaf(), i1.is_leaf()

        branch = log_p + log_2 + log_la

        if not is_tip0 and not is_tip1:
            node.add_feature('lx', branch + getattr(i0, 'lx') + getattr(i1, 'lx'))
            return

        if is_tip0 and is_tip1:
            log_lnx0, log_lnn0 = get_log_lnx_lnn_tip(i0, i1)
            log_lnx1, log_lnn1 = get_log_lnx_lnn_tip(i1, i0)
            node.add_feature('lx', branch + log_sum([getattr(i0, 'lxx') + getattr(i1, 'lxx'),
                                                     getattr(i0, 'lxn') + log_lnx1,
                                                     log_lnx0 + getattr(i1, 'lxn'),
                                                     log_lnn0 + log_lnn1]))
            return

        # i0 is a tip and i1 is internal
        if is_tip1:
            i0, i1 = i1, i0
        node.add_feature('lx', branch + log_sum([getattr(i0, 'lxx') + getattr(i1, 'lx'),
                                                 getattr(i0, 'lxn') + get_log_ln_internal(i1, i0)]))

    def get_log_lnx_lnn_tip(tip, notifier):
        """
        Calculates loglikelihood densities of a partner tip branch, who did or did not notify further,
        given the tip's notifier.

        :param tip: tip node corresponding to the partner
        :param notifier: tip node corresponding to the tip's notifier
        :return: loglikelihood densities
        """
        ti = getattr(tip, TIME)
        tr = getattr(notifier, TIME)
        tj = ti - tip.dist

        if tr < tj:
            return -np.inf
        if tr > ti:
            return getattr(tip, 'lnx_late'), getattr(tip, 'lnn_late')
        else:
            log_po = get_log_po(la=la, psi=psi, c1=c1, t=tj, ti=tr, E1=get_E1(c1=c1, c2=c2, t=tj, T=T),
                                E2=get_E2(c1=c1, c2=c2, ti=tr, T=T))
            log_pn = get_log_pn(la=la, psi=psi, t=tj, ti=tr)
            log_pp = get_log_pp(la=la, phi=phi, t=tr, ti=ti)
            return log_po + log_pp + log_phi + log_not_ups, log_pn + log_pp + log_phi + log_ups

    def get_log_ln_internal(node, notifier):
        """
        Calculates loglikelihood density of a partner node branch.

        :param node: partner node
        :param notifier: tip node corresponding to the node's notifier
        :return: loglikelihood density
        """
        ti = getattr(node, TIME)
        tr = getattr(notifier, TIME)

        # if the partner is notified they stop transmitting, hence must be notified after ti
        if tr < ti:
            return -np.inf

        i0, i1 = node.children
        is_tip0, is_tip1 = i0.is_leaf(), i1.is_leaf()

        branch = getattr(node, 'logpo') + log_la

        if not is_tip0 and not is_tip1:
            return branch + log_sum([get_log_ln_internal(i0, notifier) + getattr(i1, 'lx'),
                                     getattr(i0, 'lx') + get_log_ln_internal(i1, notifier)])

        ti0, ti1 = getattr(i0, TIME), getattr(i1, TIME)

        if is_tip0 and is_tip1:
            log_lnx_0_by_1, log_lnn_0_by_1 = get_log_lnx_lnn_tip(i0, i1)
            log_lnx_0_by_r, log_lnn_0_by_r = get_log_lnx_lnn_tip(i0, notifier)
            log_lnx_0_by_both, log_lnn_0_by_both = (log_lnx_0_by_1, log_lnn_0_by_1) if ti1 < tr \
                else (log_lnx_0_by_r, log_lnn_0_by_r)
            log_lnx_1_by_0, log_lnn_1_by_0 = get_log_lnx_lnn_tip(i1, i0)
            log_lnx_1_by_r, log_lnn_1_by_r = get_log_lnx_lnn_tip(i1, notifier)
            log_lnx_1_by_both, log_lnn_1_by_both = (log_lnx_1_by_0, log_lnn_1_by_0) if ti0 < tr \
                else (log_lnx_1_by_r, log_lnn_1_by_r)

            return branch + log_sum([log_lnx_0_by_both + getattr(i1, 'lxn'),
                                     log_lnn_0_by_both + log_lnn_1_by_0,
                                     getattr(i0, 'lxn') + log_lnx_1_by_both,
                                     log_lnn_0_by_1 + log_lnn_1_by_both,
                                     log_lnn_0_by_r + log_lnx_1_by_0,
                                     log_lnx_0_by_1 + log_lnn_1_by_r,
                                     log_lnx_0_by_r + getattr(i1, 'lxx'),
                                     getattr(i0, 'lxx') + log_lnx_1_by_r])

        # i0 is a tip and i1 is internal
        if is_tip1:
            i0, i1 = i1, i0
            ti0, ti1 = ti1, ti0
        log_lnx_0_by_r, log_lnn_0_by_r = get_log_lnx_lnn_tip(i0, notifier)
        return branch + log_sum([log_lnn_0_by_r + get_log_ln_internal(i1, i0),
                                 log_lnx_0_by_r + getattr(i1, 'lx'),
                                 getattr(i0, 'lxn') + get_log_ln_internal(i1, i0 if ti0 < tr else notifier),
                                 getattr(i0, 'lxx') + get_log_ln_internal(i1, notifier)])

    def process_tree(tree):
        for node in tree.traverse('postorder'):
            process_node(node)
        return getattr(tree, 'lx')

    if threads > 1 and len(forest) > 1:
        with ThreadPool(processes=threads) as pool:
            result = sum(pool.map(func=process_tree, iterable=forest, chunksize=max(1, len(forest) // threads + 1)))
    else:
        result = sum(process_tree(tree) for tree in forest)

    u = get_u(la, psi, c1, E1=get_E1(c1=c1, c2=c2, t=0, T=T))
    result += len(forest) * u / (1 - u) * np.log(u)
    # print(la, psi, phi, rho, upsilon, '-->', result)
    return result


def get_T(T, forest):
    if T is None:
        T = 0
        for tree in forest:
            T = max(T, max(getattr(_, TIME) for _ in tree))
    return T


def save_results(vs, cis, log, ci=False):
    os.makedirs(os.path.dirname(os.path.abspath(log)), exist_ok=True)
    with open(log, 'w+') as f:
        f.write(',{}\n'.format(','.join(['R0', 'infectious time', 'sampling probability', 'notification probability',
                                         'removal time after notification',
                                         'transmission rate', 'removal rate', 'partner removal rate'])))
        la, psi, phi, rho, rho_p = vs
        R0 = la / psi
        rt = 1 / psi
        prt = 1 / phi
        (la_min, la_max), (psi_min, psi_max), (psi_p_min, psi_p_max), (rho_min, rho_max), (rho_p_min, rho_p_max) = cis
        R0_min, R0_max = la_min / psi, la_max / psi
        rt_min, rt_max = 1 / psi_max, 1 / psi_min
        prt_min, prt_max = 1 / psi_p_max, 1 / psi_p_min
        f.write('value,{}\n'.format(','.join(str(_) for _ in [R0, rt, rho, rho_p, prt, la, psi, phi])))
        if ci:
            f.write('CI_min,{}\n'.format(
                ','.join(str(_) for _ in [R0_min, rt_min, rho_min, rho_p_min, prt_min, la_min, psi_min, psi_p_min])))
            f.write('CI_max,{}\n'.format(
                ','.join(str(_) for _ in [R0_max, rt_max, rho_max, rho_p_max, prt_max, la_max, psi_max, psi_p_max])))


def main():
    """
    Entry point for tree parameter estimation with the BDPN model with command-line arguments.
    :return: void
    """
    import argparse

    parser = \
        argparse.ArgumentParser(description="Estimated BDPN parameters.")
    parser.add_argument('--la', required=False, default=None, type=float, help="transmission rate")
    parser.add_argument('--psi', required=False, default=None, type=float, help="removal rate")
    parser.add_argument('--p', required=False, default=None, type=float, help='sampling probability')
    parser.add_argument('--pn', required=False, default=None, type=float, help='notification probability')
    parser.add_argument('--phi', required=False, default=None, type=float, help='partner removal rate')
    parser.add_argument('--log', required=True, type=str, help="output log file")
    parser.add_argument('--nwk', required=True, type=str, help="input tree file")
    parser.add_argument('--upper_bounds', required=False, type=float, nargs=5,
                        help="upper bounds for parameters (la, psi, phi, p, pn)", default=DEFAULT_UPPER_BOUNDS)
    parser.add_argument('--lower_bounds', required=False, type=float, nargs=5,
                        help="lower bounds for parameters (la, psi, phi, p, pn)", default=DEFAULT_LOWER_BOUNDS)
    parser.add_argument('--ci', action="store_true", help="calculate the CIs")
    params = parser.parse_args()

    # if os.path.exists(params.nwk.replace('.nwk', '.log')):
    #     df = pd.read_csv(params.nwk.replace('.nwk', '.log'))
    #     R, it, p, pn, rt = df.iloc[0, :5]
    #     psi = 1 / it
    #     la = R * psi
    #     phi = 1 / rt
    #     print('Real parameters: ', np.array([la, psi, phi, p, pn]))
    #     params.p = p

    if params.la is None and params.psi is None and params.p is None:
        raise ValueError('At least one of the BD model parameters (la, psi, p) needs to be specified '
                         'for identifiability')

    forest = read_forest(params.nwk)
    print('Read a forest of {} trees with {} tips in total'.format(len(forest), sum(len(_) for _ in forest)))
    vs, cis = infer(forest, **vars(params))

    save_results(vs, cis, params.log, ci=params.ci)


def infer(forest, la=None, psi=None, phi=None, p=None, pn=None,
          lower_bounds=DEFAULT_LOWER_BOUNDS, upper_bounds=DEFAULT_UPPER_BOUNDS, ci=False, **kwargs):
    """
    Infers BDPN model parameters from a given forest.

    :param forest: list of one or more trees
    :param la: transmission rate
    :param psi: removal rate
    :param phi: partner removal rate
    :param p: sampling probability
    :param pn: partner notification probability
    :param lower_bounds: array of lower bounds for parameter values (la, psi, partner_psi, p, pn)
    :param upper_bounds: array of upper bounds for parameter values (la, psi, partner_psi, p, pn)
    :param ci: whether to calculate the CIs or not
    :return: tuple(vs, cis) of estimated parameter values vs=[la, psi, partner_psi, p, pn]
        and CIs ci=[[la_min, la_max], ..., [pn_min, pn_max]]. In the case when CIs were not set to be calculated,
        their values would correspond exactly to the parameter values.
    """
    if la is None and psi is None and p is None:
        raise ValueError('At least one of the BD model parameters (la, psi, p) needs to be specified '
                         'for identifiability')
    bounds = np.zeros((5, 2), dtype=np.float64)
    bounds[:, 0] = lower_bounds
    bounds[:, 1] = upper_bounds
    vs, _ = bd_model.infer(forest, la=la, psi=psi, p=p,
                           lower_bounds=bounds[[0, 1, 3], 0], upper_bounds=bounds[[0, 1, 3], 1], ci=False)
    start_parameters = np.array([vs[0], vs[1], vs[1] * 10 if phi is None or phi < 0 else phi,
                                 vs[-1], 0.5 if pn is None or pn <= 0 or pn > 1 else pn])
    input_params = np.array([la, psi, phi, p, pn])
    print('Input parameters {} are fixed: {}'.format(PARAMETER_NAMES[input_params != None],
                                                     input_params[input_params != None]))
    print('Starting BDPN parameters: {}'.format(start_parameters))
    vs, cis = optimize_likelihood_params(forest, input_parameters=input_params,
                                         loglikelihood=loglikelihood, bounds=bounds[input_params == None],
                                         start_parameters=start_parameters, cis=ci)
    print('Estimated BDPN parameters: {}'.format(vs))
    return vs, cis


if '__main__' == __name__:
    main()
