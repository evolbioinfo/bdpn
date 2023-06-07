import os

import numpy as np
from ete3 import Tree

from bdpn.bd import get_c1, get_c3, get_log_p, get_u
from bdpn.parameter_estimator import get_rough_rate_etimates, optimize_likelihood_params, rescale_log
from bdpn.tree_manager import TIME, annotate_tree


def get_log_p_o(t, la, psi, rho, T, ti):
    c1 = get_c1(la, psi, rho)
    c3 = get_c3(la, psi, rho, T, ti, c1=c1)
    # c2 = get_c2(la, psi, rho, T, c1=c1)
    # return np.log(c3) + (1/2 * (la + psi + c1) * (t - ti)) - np.log(c2 * np.exp(c1 * t) + 1)
    # Inline c2 to avoid underflow
    return np.log(c3) + (1 / 2 * (la + psi + c1) * (t - ti)) \
        - np.log((c1 + la - psi) / (c1 - la + psi) * np.exp(c1 * (t - T)) + 1)

def get_log_p_nh(t, la, psi, ti):
    # return np.exp((la + psi) * (t - ti))
    return (la + psi) * (t - ti)


def loglikelihood(tree, la, psi, psi_n, rho, rho_n, T=None):
    if not hasattr(tree, TIME):
        annotate_tree(tree)
    if T is None:
        T = max(getattr(_, TIME) for _ in tree)

    for n in tree.traverse():
        ti = getattr(n, TIME)
        t = ti - n.dist
        n.add_feature('p', get_log_p(t, la, psi, rho, T, ti))
        n.add_feature('po', get_log_p_o(t, la, psi, rho, T, ti))
        if n.is_leaf():
            n.add_feature('pnh', get_log_p_nh(t, la, psi, ti))

    log_sampled = np.log(psi) + np.log(rho)
    log_sampled_and_notified = log_sampled + np.log(rho_n)
    log_sampled_not_notified = log_sampled + np.log(1 - rho_n)
    for n in tree.traverse('postorder'):
        ti = getattr(n, TIME)
        t = ti - n.dist
        if n.is_leaf():
            n.add_feature('LU', getattr(n, 'p') + log_sampled_not_notified)
            n.add_feature('LN', getattr(n, 'pnh') + log_sampled_and_notified)
            n.add_feature('CP', [(t, ti, 1)])
            continue
            
        c1, c2 = n.children
        result = []
        po_log_la = getattr(n, 'po') + np.log(la)
        po_log_la_lu2 = po_log_la + getattr(c2, 'LU')
        po_log_la_lu1 = po_log_la + getattr(c1, 'LU')
        for (t, ti, c) in getattr(c1, 'CP'):
            result.append((t, ti, po_log_la_lu2 + c))
        for (t, ti, c) in getattr(c2, 'CP'):
            result.append((t, ti, po_log_la_lu1 + c))
        n.add_feature('CP', result)
            
        log_p_2la = getattr(n, 'p') + np.log(2 * la)
        log_lulu = getattr(c1, 'LU') + getattr(c2, 'LU')
        if not c1.is_leaf() and not c2.is_leaf():
            n.add_feature('LU', log_p_2la + log_lulu)
            continue
        result = [log_lulu]
        log_psi_n = np.log(psi_n)
        if c1.is_leaf():
            tn = getattr(c1, TIME)
            log_ln1 = getattr(c1, 'LN')
            for (t, ti, c) in getattr(c2, 'CP'):
                if t <= tn <= ti:
                    result.append(log_ln1 + c + get_log_p_o(t, la, psi, rho, T, tn) + (-psi_n * (ti - tn)) + log_psi_n)
        if c2.is_leaf():
            tn = getattr(c2, TIME)
            log_ln2 = getattr(c2, 'LN')
            for (t, ti, c) in getattr(c1, 'CP'):
                if t <= tn <= ti:
                    result.append(log_ln2 + c + get_log_p_o(t, la, psi, rho, T, tn) + (-psi_n * (ti - tn)) + log_psi_n)
        if len(result) == 1:
            n.add_feature('LU', log_p_2la + log_lulu)
        else:
            result = np.array(result, dtype=np.float64)
            factors = rescale_log(result)
            n.add_feature('LU', log_p_2la + np.log(np.sum(np.exp(result))) - factors)

    u = get_u(0, la, psi, rho, T)
    return getattr(tree, 'LU') + u / (1 - u) * np.log(u)
    # return getattr(tree, 'LU')


def get_bounds_start(tree, la, psi, psi_n, rho, rho_n):
    bounds = []

    l_ext = []
    l_int = []
    for n in tree.traverse('preorder'):
        if n.dist:
            (l_int if not n.is_leaf() else l_ext).append(n.dist)
    min_la_rate = 1 / np.max(l_int)
    max_la_rate = 1 / np.percentile(l_int, 0.1)
    min_psi_rate = 1 / np.max(l_ext)
    max_psi_rate = 1 / np.percentile(l_ext, 0.1)
    if la is None:
        bounds.append([min_la_rate, max_la_rate])
    if psi is None:
        bounds.append([min_psi_rate, max_psi_rate])
    if psi_n is None:
        bounds.append([min_psi_rate, max_psi_rate])
    bounds.extend([[1e-3, 1]] * int(rho is None))
    bounds.extend([[0, 1]] * int(rho_n is None))
    avg_la_rate = 1 / np.median(l_int)
    avg_psi_rate = 1 / np.median(l_ext)
    return np.array(bounds, np.float64), [avg_la_rate, avg_psi_rate, (avg_psi_rate + max_psi_rate) / 2, 0.5, 0.5]


    # avg_rate, max_rate, min_rate = get_rough_rate_etimates(tree)
    # bounds.extend([[min_rate, max_rate]] * (int(la is None) + int(psi is None)))
    # if psi_n is None:
    #     bounds.extend([[min_rate, np.inf]])
    # bounds.extend([[1e-3, 1]] * int(rho is None))
    # bounds.extend([[0, 1]] * int(rho_n is None))
    # return np.array(bounds, np.float64), [avg_rate, avg_rate, avg_rate, 0.5, 0.5]


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
    parser.add_argument('--upper_bounds', required=True, type=float, nargs='+', help="upper bounds for parameters")
    parser.add_argument('--lower_bounds', required=True, type=float, nargs='+', help="lower bounds for parameters")
    params = parser.parse_args()

    if params.la is None and params.psi is None and params.p is None and params.pn is None and params.partner_psi is None:
        raise ValueError('At least one of the model parameters needs to be specified for identifiability')

    tree = Tree(params.nwk)
    bounds = np.zeros((5, 2), dtype=np.float64)
    bounds[:, 0] = params.lower_bounds
    bounds[:, 1] = params.upper_bounds
    start_parameters = (bounds[:, 0] + bounds[:, 1]) / 2
    input_params = np.array([params.la, params.psi, params.partner_psi, params.p, params.pn])
    vs = optimize_likelihood_params(tree, input_parameters=input_params,
                                    loglikelihood=loglikelihood, bounds=bounds[input_params == None],
                                    start_parameters=start_parameters)

    os.makedirs(os.path.dirname(os.path.abspath(params.log)), exist_ok=True)
    with open(params.log, 'w+') as f:
        f.write('{}\n'.format(','.join(['R0', 'infectious time', 'sampling probability', 'notification probability',
                                        'removal time after notification'])))
        f.write('{}\n'.format(','.join(str(_) for _ in [vs[0] / vs[1], 1 / vs[1], vs[3], vs[4], 1 / vs[2]])))


if '__main__' == __name__:
    main()

