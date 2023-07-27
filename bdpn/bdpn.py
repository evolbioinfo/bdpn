import os
from multiprocessing.pool import ThreadPool

import numpy as np
from pastml.tree import read_forest

from bdpn.parameter_estimator import optimize_likelihood_params, rescale_log
from bdpn.tree_manager import TIME, annotate_tree
from bdpn.formulas import get_log_p, get_c1, get_c2, get_E2, get_E1, get_log_po, get_log_pnh, get_log_po_from_p_pnh, \
    get_u


def loglikelihood(forest, la, psi, psi_n, rho, rho_n, T=None, threads=1):
    for tree in forest:
        if not hasattr(tree, TIME):
            annotate_tree(tree)
    if T is None:
        T = 0
        for tree in forest:
            T = max(T, max(getattr(_, TIME) for _ in tree))

    c1 = get_c1(la=la, psi=psi, rho=rho)
    c2 = get_c2(la=la, psi=psi, c1=c1)

    nodes = []
    for tree in forest:
        for n in tree.traverse():
            nodes.append(n)

    def annotate_node(n):
        ti = getattr(n, TIME)
        t = ti - n.dist

        E1 = get_E1(c1=c1, c2=c2, t=t, T=T)
        E2 = get_E2(c1=c1, c2=c2, ti=ti, T=T)

        log_p = get_log_p(c1=c1, t=t, ti=ti, E1=E1, E2=E2)
        log_pnh = get_log_pnh(la=la, psi=psi, t=t, ti=ti)
        log_po = get_log_po_from_p_pnh(log_p=log_p, log_pnh=log_pnh)

        n.add_feature('p', log_p)
        n.add_feature('po', log_po)
        if n.is_leaf():
            n.add_feature('pnh', log_pnh)

    def get_lpn_tip(partner, notifier):
        """
        Calculates loglikelihood density of a partner tip branch, who notified further, given their notifier.

        :param partner: tip node corresponding to the partner
        :param notifier: tip node corresponding to the notifier
        :return: loglikelihood density
        """
        ti = getattr(partner, TIME)
        tr = getattr(notifier, TIME)
        t = ti - partner.dist
        if tr < t:
            return -np.inf
        if tr > ti:
            return getattr(partner, 'LP-late-n')
        else:
            return getattr(partner, 'LP-n') + get_log_pnh(la, psi, t, tr) - psi_n * (ti - tr)

    def get_lpnn_tip(partner, notifier):
        """
        Calculates loglikelihood density of a partner tip branch, who did not notify further, given their notifier.

        :param partner: tip node corresponding to the partner
        :param notifier: tip node corresponding to the notifier
        :return: loglikelihood density
        """
        ti = getattr(partner, TIME)
        tr = getattr(notifier, TIME)
        t = ti - partner.dist
        if tr < t:
            return -np.inf
        if tr > ti:
            return getattr(partner, 'LP-late-nn')
        else:
            E1 = get_E1(c1=c1, c2=c2, t=t, T=T)
            E2 = get_E2(c1=c1, c2=c2, ti=tr, T=T)
            return getattr(partner, 'LP-nn') + get_log_po(la, psi, c1, t=t, ti=tr, E1=E1, E2=E2) - psi_n * (ti - tr)

    log_la = np.log(la)
    log_2la = np.log(2) + log_la
    log_psi = np.log(psi)
    log_psi_n = np.log(psi_n)
    log_rho = np.log(rho)
    log_rho_n = np.log(rho_n)
    log_not_rho_n = np.log(1 - rho_n)

    def get_lp_internal(partner, notifier):
        """
        Calculates loglikelihood density of a partner internal branch and its subtree, given their notifier.

        :param partner: internal node corresponding to the partner
        :param notifier: tip node corresponding to the notifier
        :return: loglikelihood density
        """
        ti = getattr(partner, TIME)
        tr = getattr(notifier, TIME)
        t = ti - partner.dist
        if tr < t:
            return -np.inf

        child_1, child_2 = partner.children
        is_tip_1 = child_1.is_leaf()
        is_tip_2 = child_2.is_leaf()
        if not is_tip_1 and not is_tip_2:
            result = [
                # the partner is somewhere in child 1's subtree
                get_lp_internal(child_1, notifier) + getattr(child_2, 'LU'),
                # the partner is somewhere in child 2's subtree
                get_lp_internal(child_2, notifier) + getattr(child_1, 'LU')
            ]
        elif is_tip_1 and is_tip_2:
            t1 = getattr(child_1, TIME)
            t2 = getattr(child_2, TIME)

            lpn_child1_by_child2 = get_lpn_tip(child_1, child_2)
            lpn_child1_by_notifier = get_lpn_tip(child_1, notifier)
            lpn_child1_by_first_notifier_child2 = lpn_child1_by_notifier if tr <= t2 else lpn_child1_by_child2

            lpn_child2_by_child1 = get_lpn_tip(child_2, child_1)
            lpn_child2_by_notifier = get_lpn_tip(child_2, notifier)
            lpn_child2_by_first_notifier_child1 = lpn_child2_by_notifier if tr <= t1 else lpn_child2_by_child1

            lpnn_child1_by_child2 = get_lpnn_tip(child_1, child_2)
            lpnn_child1_by_notifier = get_lpnn_tip(child_1, notifier)
            lpnn_child1_by_first_notifier_child2 = lpnn_child1_by_notifier if tr <= t2 else lpnn_child1_by_child2

            lpnn_child2_by_child1 = get_lpnn_tip(child_2, child_1)
            lpnn_child2_by_notifier = get_lpnn_tip(child_2, notifier)
            lpnn_child2_by_first_notifier_child1 = lpnn_child2_by_notifier if tr <= t1 else lpnn_child2_by_child1

            result = [
                # child 1 was notified by both child 2 and the notifier, and did not notify child 2
                lpnn_child1_by_first_notifier_child2 + getattr(child_2, 'LU-n'),
                # child 1 was notified by both child 2 and the notifier, and also notified child 2
                lpn_child1_by_first_notifier_child2 + lpn_child2_by_child1,
                # child 1 was notified only by the notifier, and notified child 2
                lpn_child1_by_notifier + lpnn_child2_by_child1,
                # child 1 was notified only by the notifier, and did not notify child 2
                lpnn_child1_by_notifier + getattr(child_2, 'LU-nn'),
                # child 2 was notified by both child 1 and the notifier, and did not notify child 1
                lpnn_child2_by_first_notifier_child1 + getattr(child_1, 'LU-n'),
                # child 2 was notified by both child 1 and the notifier, and also notified child 1
                lpn_child2_by_first_notifier_child1 + lpn_child1_by_child2,
                # child 2 was notified only by the notifier, and notified child 1
                lpn_child2_by_notifier + lpnn_child1_by_child2,
                # child 2 was notified only by the notifier, and did not notify child 1
                lpnn_child2_by_notifier + getattr(child_1, 'LU-nn')
            ]
        elif is_tip_1 and not is_tip_2:
            t1 = getattr(child_1, TIME)

            lp_child2_by_child1 = get_lp_internal(child_2, child_1)
            lp_child2_by_notifier = get_lp_internal(child_2, notifier)
            lp_child2_by_first_notifier_child1 = lp_child2_by_notifier if tr <= t1 else lp_child2_by_child1

            result = [
                # child 1 was notified by the notifier and notified child 2
                get_lpn_tip(child_1, notifier) + lp_child2_by_child1,
                # child 1 was notified by the notifier and did not notify child 2
                get_lpnn_tip(child_1, notifier) + getattr(child_2, 'LU'),
                # child 2 was notified by both child 1 and the notifier
                lp_child2_by_first_notifier_child1 + getattr(child_1, 'LU-n'),
                # child 2 was notified only by the notifier
                lp_child2_by_notifier + getattr(child_1, 'LU-nn')
            ]
        elif is_tip_2 and not is_tip_1:
            t2 = getattr(child_2, TIME)

            lp_child1_by_child2 = get_lp_internal(child_1, child_2)
            lp_child1_by_notifier = get_lp_internal(child_1, notifier)
            lp_child1_by_first_notifier_child2 = lp_child1_by_notifier if tr <= t2 else lp_child1_by_child2

            result = [
                # child 2 was notified by the notifier and notified child 1
                get_lpn_tip(child_2, notifier) + lp_child1_by_child2,
                # child 2 was notified by the notifier and did not notify child 1
                get_lpnn_tip(child_2, notifier) + getattr(child_1, 'LU'),
                # child 1 was notified by both child 2 and the notifier
                lp_child1_by_first_notifier_child2 + getattr(child_2, 'LU-n'),
                # child 1 was notified only by the notifier
                lp_child1_by_notifier + getattr(child_2, 'LU-nn')
            ]

        result = np.array(result, dtype=np.float64)
        factors = rescale_log(result)
        return getattr(partner, 'po') + log_la + np.log(np.sum(np.exp(result))) - factors

    log_sampled = log_psi + log_rho
    log_sampled_and_notified = log_sampled + log_rho_n
    log_sampled_not_notified = log_sampled + log_not_rho_n
    log_partner_sampled_and_notified = log_psi_n + log_rho_n
    log_partner_sampled_not_notified = log_psi_n + log_not_rho_n

    def process_tree(tree):
        for n in tree.traverse('postorder'):
            if n.is_leaf():
                # non-notified non-notifier
                n.add_feature('LU-nn', getattr(n, 'p') + log_sampled_not_notified)
                lun = getattr(n, 'pnh') + log_sampled_and_notified
                # non-notified notifier
                n.add_feature('LU-n', lun)
                # partner who got notified too late (after being sampled via the normal procedure), and who notified
                n.add_feature('LP-late-n', lun)
                # notified partner who notified further
                n.add_feature('LP-n', log_partner_sampled_and_notified)
                # partner who got notified too late (after being sampled via the normal procedure), and did not notify
                n.add_feature('LP-late-nn', getattr(n, 'po') + log_sampled_not_notified)
                # notified partner who did not notify further
                n.add_feature('LP-nn', log_partner_sampled_not_notified)
                continue

            child_1, child_2 = n.children
            is_tip_1 = child_1.is_leaf()
            is_tip_2 = child_2.is_leaf()

            log_p_2la = getattr(n, 'p') + log_2la

            if not is_tip_1 and not is_tip_2:
                # unnotified internal branch with both children internal and hence unnotified
                result = getattr(child_1, 'LU') + getattr(child_2, 'LU')
            else:
                if is_tip_1 and is_tip_2:
                    # unnotified internal branch with both children being tips
                    result = [
                        # no notification
                        getattr(child_1, 'LU-nn') + getattr(child_2, 'LU-nn'),
                        # child 1 notified child 2
                        getattr(child_1, 'LU-n') + get_lpnn_tip(child_2, child_1),
                        # child 2 notified child 1
                        getattr(child_2, 'LU-n') + get_lpnn_tip(child_1, child_2),
                        # children notified each other
                        get_lpn_tip(child_1, child_2) + get_lpn_tip(child_2, child_1)
                    ]
                elif is_tip_1 and not is_tip_2:
                    # unnotified internal branch with child 1 being a tip and child 2 internal
                    result = [
                        # no notification
                        getattr(child_1, 'LU-nn') + getattr(child_2, 'LU'),
                        # child 1 notified someone in child 2's subtree
                        getattr(child_1, 'LU-n') + get_lp_internal(child_2, child_1)
                    ]
                elif is_tip_2 and not is_tip_1:
                    # unnotified internal branch with child 2 being a tip and child 1 internal
                    result = [
                        # no notification
                        getattr(child_2, 'LU-nn') + getattr(child_1, 'LU'),
                        # child 2 notified someone in child 1's subtree
                        getattr(child_2, 'LU-n') + get_lp_internal(child_1, child_2)
                    ]
                result = np.array(result, dtype=np.float64)
                factors = rescale_log(result)
                result = np.log(np.sum(np.exp(result))) - factors
            n.add_feature('LU', log_p_2la + result)

        return getattr(tree, 'LU' if not tree.is_leaf() else 'LU-nn')

    if threads > 1:
        with ThreadPool(processes=threads) as pool:
            pool.map(func=annotate_node, iterable=nodes, chunksize=max(1, len(nodes) // threads + 1))
            if len(forest) > 1:
                result = sum(pool.map(func=process_tree, iterable=forest, chunksize=max(1, len(forest) // threads + 1)))
            else:
                result = sum(process_tree(tree) for tree in forest)
    else:
        for n in nodes:
            annotate_node(n)
        result = sum(process_tree(tree) for tree in forest)

    u = get_u(la, psi, c1, E1=get_E1(c1=c1, c2=c2, t=0, T=T))
    return result + len(forest) * u / (1 - u) * np.log(u)


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
    parser.add_argument('--ci', action="store_true", help="calculate the CIs")
    params = parser.parse_args()

    if params.la is None and params.psi is None and params.p is None and params.pn is None and params.partner_psi is None:
        raise ValueError('At least one of the model parameters needs to be specified for identifiability')

    forest = read_forest(params.nwk)
    print('Read a forest of {} trees with {} tips in total'.format(len(forest), sum(len(_) for _ in forest)))
    bounds = np.zeros((5, 2), dtype=np.float64)
    bounds[:, 0] = params.lower_bounds
    bounds[:, 1] = params.upper_bounds
    start_parameters = (bounds[:, 0] + bounds[:, 1]) / 2
    input_params = np.array([params.la, params.psi, params.partner_psi, params.p, params.pn])
    vs, cis = optimize_likelihood_params(forest, input_parameters=input_params,
                                         loglikelihood=loglikelihood, bounds=bounds[input_params == None],
                                         start_parameters=start_parameters, cis=params.ci)

    os.makedirs(os.path.dirname(os.path.abspath(params.log)), exist_ok=True)
    with open(params.log, 'w+') as f:
        f.write('\t{}\n'.format(','.join(['R0', 'infectious time', 'sampling probability', 'notification probability',
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
        if params.ci:
            f.write('CI_min,{}\n'.format(
                ','.join(str(_) for _ in [R0_min, rt_min, rho_min, rho_p_min, prt_min, la_min, psi_min, psi_p_min])))
            f.write('CI_max,{}\n'.format(
                ','.join(str(_) for _ in [R0_max, rt_max, rho_max, rho_p_max, prt_max, la_max, psi_max, psi_p_max])))


if '__main__' == __name__:
    main()
