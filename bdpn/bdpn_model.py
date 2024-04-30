import os
from multiprocessing.pool import ThreadPool

import numpy as np

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




def loglikelihood(forest, la, psi, phi, rho, upsilon, T=None, threads=1):
    annotate_forest_with_time(forest)
    if T is None:
        T = 0
        for tree in forest:
            T = max(T, max(getattr(_, TIME) for _ in tree))

    c1 = get_c1(la=la, psi=psi, rho=rho)
    c2 = get_c2(la=la, psi=psi, c1=c1)

    log_psi = np.log(psi)
    log_phi = np.log(phi)
    log_rho = np.log(rho)
    log_ups = np.log(upsilon)
    log_not_ups = np.log(1 - upsilon)
    log_2 = np.log(2)
    log_la = np.log(la)

    def process_node(node):
        ti = getattr(node, TIME)
        tj = ti - node.dist

        E1 = get_E1(c1=c1, c2=c2, t=tj, T=T)
        E2 = get_E2(c1=c1, c2=c2, ti=ti, T=T)

        log_p = get_log_p(c1=c1, t=tj, ti=ti, E1=E1, E2=E2)
        log_po = get_log_po_from_p_pn(log_p=log_p, log_pn=log_pn)

        if node.is_leaf():
            log_pn = get_log_pn(la=la, psi=psi, t=tj, ti=ti)
            node.add_feature('lxx', log_p + log_psi + log_rho + log_not_ups)
            node.add_feature('lxn', log_pn + log_psi + log_rho + log_ups)
            node.add_feature('lnx_late', log_po + log_psi + log_rho + log_not_ups)
            node.add_feature('lnn_late', log_pn + log_psi + log_rho + log_ups)
            return

        node.add_feature('logp', log_p)
        node.add_feature('logpo', log_po)

        child_1, child_2 = node.children
        is_tip_1 = child_1.is_leaf()
        is_tip_2 = child_2.is_leaf()

        branch = log_p + log_2 + log_la

        if not is_tip_1 and not is_tip_2:
            node.add_feature('lx', branch + getattr(child_1, 'lx') + getattr(child_2, 'lx'))
            return
        if is_tip_1 and is_tip_2:
            result = [getattr(child_1, 'lxx') + getattr(child_2, 'lxx'),
                      getattr(child_1, 'lxn') + get_log_lnx_tip(child_2, child_1),
                      get_log_lnx_tip(child_1, child_2) + getattr(child_2, 'lxn'),
                      get_log_lnn_tip(child_1, child_2), get_log_lnn_tip(child_2, child_1)]
            result = np.array(result, dtype=np.float64)
            factors = rescale_log(result)
            node.add_feature('lx', branch + np.log(np.sum(np.exp(result))) - factors)
            return
        # child 1 is a tip and child 2 is internal
        if is_tip_2:
            child_1, child_2 = child_2, child_1
        result = [getattr(child_1, 'lxx') + getattr(child_2, 'lx'),
                  getattr(child_1, 'lxn') + get_log_ln_internal(child_2, child_1)]
        result = np.array(result, dtype=np.float64)
        factors = rescale_log(result)
        node.add_feature('lx', branch + np.log(np.sum(np.exp(result))) - factors)

    def get_log_lxx_tip(tip):
        """
        Calculates loglikelihood density of an unnotified tip branch, who did not notify.

        :param tip: tip node
        :return: loglikelihood density
        """
        return getattr(tip, 'logp') + log_psi + log_rho + log_not_ups

    def get_log_lxn_tip(tip):
        """
        Calculates loglikelihood density of an unnotified tip branch, who notified their partner.

        :param tip: tip node
        :return: loglikelihood density
        """
        return getattr(tip, 'logpn') + log_psi + log_rho + log_ups

    def get_log_lnx_tip(tip, notifier):
        """
        Calculates loglikelihood density of a partner tip branch, who did not notify further, given the tip's notifier.

        :param tip: tip node corresponding to the partner
        :param notifier: tip node corresponding to the tip's notifier
        :return: loglikelihood density
        """
        ti = getattr(tip, TIME)
        tr = getattr(notifier, TIME)
        tj = ti - tip.dist

        if tr < tj:
            return -np.inf
        if tr > ti:
            return getattr(tip, 'lnx_late')
            # return getattr(tip, 'logpo') + log_psi + log_rho + log_not_ups
        else:
            return get_log_po(la=la, psi=psi, c1=c1, t=tj, ti=tr,
                              E1=get_E1(c1=c1, c2=c2, t=tj, T=T),
                              E2=get_E2(c1=c1, c2=c2, ti=tr, T=T)) \
                + get_log_pp(la=la, phi=phi, t=tr, ti=ti) \
                + log_phi + log_not_ups

    def get_log_lnn_tip(tip, notifier):
        """
        Calculates loglikelihood density of a partner tip branch, who notified further, given the tip's notifier.

        :param tip: tip node corresponding to the partner
        :param notifier: tip node corresponding to the tip's notifier
        :return: loglikelihood density
        """
        ti = getattr(tip, TIME)
        tr = getattr(notifier, TIME)
        tj = ti - tip.dist

        if tr < tj:
            return -np.inf
        if tr > ti:
            return getattr(tip, 'lnn_late')
            # return getattr(tip, 'logpn') + log_psi + log_rho + log_ups
        else:
            return get_log_pn(la=la, psi=psi, t=tj, ti=tr) \
                + get_log_pp(la=la, phi=phi, t=tr, ti=ti) \
                + log_phi + log_ups

    def get_log_lx_internal(node):
        """
        Calculates loglikelihood density of an unnotified node branch.

        :param node: unnotified node
        :return: loglikelihood density
        """
        child_1, child_2 = node.children
        is_tip_1 = child_1.is_leaf()
        is_tip_2 = child_2.is_leaf()

        branch = getattr(node, 'logp') + log_2 + log_la

        if not is_tip_1 and not is_tip_2:
            return branch + get_log_lx_internal(child_1) + get_log_lx_internal(child_2)
        if is_tip_1 and is_tip_2:
            result = [get_log_lxx_tip(child_1) + get_log_lxx_tip(child_2),
                      get_log_lxn_tip(child_1) + get_log_lnx_tip(child_2, child_1),
                      get_log_lnx_tip(child_1, child_2) + get_log_lxn_tip(child_2),
                      get_log_lnn_tip(child_1, child_2), get_log_lnn_tip(child_2, child_1)]
            result = np.array(result, dtype=np.float64)
            factors = rescale_log(result)
            return branch + np.log(np.sum(np.exp(result))) - factors
        # child 1 is a tip and child 2 is internal
        if is_tip_2:
            child_1, child_2 = child_2, child_1
        result = [get_log_lxx_tip(child_1) + get_log_lx_internal(child_2),
                  get_log_lxn_tip(child_1) + get_log_ln_internal(child_2, child_1)]
        result = np.array(result, dtype=np.float64)
        factors = rescale_log(result)
        return branch + np.log(np.sum(np.exp(result))) - factors

    def get_log_ln_internal(node, notifier):
        """
        Calculates loglikelihood density of a partner node branch.

        :param node: partner node
        :param notifier: tip node corresponding to the node's notifier
        :return: loglikelihood density
        """
        ti = getattr(node, TIME)
        tr = getattr(notifier, TIME)
        tj = ti - node.dist

        if tr < tj:
            return -np.inf

        child_1, child_2 = node.children
        is_tip_1 = child_1.is_leaf()
        is_tip_2 = child_2.is_leaf()

        branch = getattr(node, 'logpo') + log_la

        if not is_tip_1 and not is_tip_2:
            result = [get_log_ln_internal(child_1, notifier) + getattr(child_2, 'lx'),
                      getattr(child_1, 'lx') + get_log_ln_internal(child_2, notifier)]
            result = np.array(result, dtype=np.float64)
            factors = rescale_log(result)
            return branch + np.log(np.sum(np.exp(result))) - factors

        ti1 = getattr(child_1, TIME)
        ti2 = getattr(child_2, TIME)

        if is_tip_1 and is_tip_2:
            first_notifier_i1 = child_1 if ti1 < tr else notifier
            first_notifier_i2 = child_2 if ti2 < tr else notifier
            result = [get_log_lnx_tip(child_1, first_notifier_i2) + getattr(child_2, 'lxn'),
                      get_log_lnn_tip(child_1, first_notifier_i2) + get_log_lnn_tip(child_2, child_1),
                      getattr(child_1, 'lxn') + get_log_lnx_tip(child_2, first_notifier_i1),
                      get_log_lnn_tip(child_1, child_2) + get_log_lnn_tip(child_2, first_notifier_i1),
                      get_log_lnn_tip(child_1, notifier) + get_log_lnx_tip(child_2, child_1),
                      get_log_lnx_tip(child_1, child_2) + get_log_lnn_tip(child_2, notifier),
                      get_log_lnx_tip(child_1, notifier) + getattr(child_2, 'lxx'),
                      getattr(child_1, 'lxx') + get_log_lnx_tip(child_2, notifier)]
            result = np.array(result, dtype=np.float64)
            factors = rescale_log(result)
            return branch + np.log(np.sum(np.exp(result))) - factors

        # child 1 is a tip and child 2 is internal
        if is_tip_2:
            child_1, child_2 = child_2, child_1
            ti1, ti2 = ti2, ti1
        first_notifier_i1 = child_1 if ti1 < tr else notifier
        result = [get_log_lnn_tip(child_1, notifier) + get_log_ln_internal(child_2, child_1),
                  get_log_lnx_tip(child_1, notifier) + getattr(child_2, 'lx'),
                  getattr(child_1, 'lxn') + get_log_ln_internal(child_2, first_notifier_i1),
                  getattr(child_1, 'lxx') + get_log_ln_internal(child_2, notifier)]
        result = np.array(result, dtype=np.float64)
        factors = rescale_log(result)
        return branch + np.log(np.sum(np.exp(result))) - factors

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
    return result + len(forest) * u / (1 - u) * np.log(u)


def save_results(vs, cis, log, ci=False):
    os.makedirs(os.path.dirname(os.path.abspath(log)), exist_ok=True)
    with open(log, 'w+') as f:
        f.write(',{}\n'.format(','.join(['R0', 'infectious time', 'sampling probability', 'notification probability',
                                          'removal time after notification',
                                          'transmission rate', 'removal rate', 'partner removal rate'])))
        la, psi, psi_p, rho, rho_p = vs
        R0 = la / psi
        rt = 1 / psi
        prt = 1 / psi_p
        (la_min, la_max), (psi_min, psi_max), (psi_p_min, psi_p_max), (rho_min, rho_max), (rho_p_min, rho_p_max) = cis
        R0_min, R0_max = la_min / psi, la_max / psi
        rt_min, rt_max = 1 / psi_max, 1 / psi_min
        prt_min, prt_max = 1 / psi_p_max, 1 / psi_p_min
        f.write('value,{}\n'.format(','.join(str(_) for _ in [R0, rt, rho, rho_p, prt, la, psi, psi_p])))
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
    parser.add_argument('--partner_psi', required=False, default=None, type=float, help='partner removal rate')
    parser.add_argument('--log', required=True, type=str, help="output log file")
    parser.add_argument('--nwk', required=True, type=str, help="input tree file")
    parser.add_argument('--upper_bounds', required=False, type=float, nargs=5,
                        help="upper bounds for parameters (la, psi, partner_psi, p, pn)", default=DEFAULT_UPPER_BOUNDS)
    parser.add_argument('--lower_bounds', required=False, type=float, nargs=5,
                        help="lower bounds for parameters (la, psi, partner_psi, p, pn)", default=DEFAULT_LOWER_BOUNDS)
    parser.add_argument('--ci', action="store_true", help="calculate the CIs")
    params = parser.parse_args()

    if params.la is None and params.psi is None and params.p is None:
        raise ValueError('At least one of the BD model parameters (la, psi, p) needs to be specified '
                         'for identifiability')

    forest = read_forest(params.nwk)
    print('Read a forest of {} trees with {} tips in total'.format(len(forest), sum(len(_) for _ in forest)))
    vs, cis = infer(forest, **vars(params))

    save_results(vs, cis, params.log, ci=params.ci)


def infer(forest, la=None, psi=None, partner_psi=None, p=None, pn=None,
          lower_bounds=DEFAULT_LOWER_BOUNDS, upper_bounds=DEFAULT_UPPER_BOUNDS, ci=False, **kwargs):
    """
    Infers BDPN model parameters from a given forest.

    :param forest: list of one or more trees
    :param la: transmission rate
    :param psi: removal rate
    :param partner_psi: partner removal rate
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
    start_parameters = np.array([vs[0], vs[1], vs[1] * 10 if partner_psi is None or partner_psi < 0 else partner_psi,
                                 vs[-1], 0.5 if pn is None or pn <= 0 or pn > 1 else pn])
    input_params = np.array([la, psi, partner_psi, p, pn])
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
