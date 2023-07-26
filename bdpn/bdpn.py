import os
from multiprocessing.pool import ThreadPool

import numpy as np
from pastml.tree import read_forest
from scipy import integrate

from bdpn.parameter_estimator import optimize_likelihood_params, rescale_log
from bdpn.tree_manager import TIME, annotate_tree
from bdpn.formulas import get_log_p, get_c1, get_c2, get_E2, get_E1, get_log_po, get_log_pnh, get_log_po_from_p_pnh, \
    get_u


# def get_log_hidden_partner_notification(t, la, psi, rho, psi_n, T, ti, c1=None, c3=None, c4=None):
#     # return np.log(
#     #     integrate.quad(lambda tau: np.exp(get_log_p(t, la, psi, rho, T, tau, c1=c1) + np.log(2 * la)
#     #                                       + get_log_p_nh(tau, la, psi, ti)
#     #                                       + get_log_p_o(tau, la, psi, rho, T, ti, c1=c1, c3=c3)
#     #                                       + (psi_n * (ti - T))), t, ti)[0])
#     # we inline everything we can in the above expression
#
#     if c4 is None:
#         c4 = (c1 + la - psi) / (c1 - la + psi)
#     la_plus_psi = la + psi
#     c5 = (3 * la_plus_psi + c1) / 2
#     non_tau_part = c1 * t - 2 * np.log(c4 * np.exp(c1 * (t - T)) + 1) \
#                    - c5 * ti \
#                    + np.log(c3) \
#                    + psi_n * (ti - T)
#
#     def get_v(tau, non_tau_part):
#         c3_p = get_c3(la, psi, rho, T, tau, c1)
#         tau_part = 2 * np.log(c3_p) + (c5 - c1) * tau \
#                    - np.log(c4 * np.exp(c1 * (tau - T)) + 1)
#         return np.exp(non_tau_part + tau_part)
#
#     return np.log(integrate.quad(lambda tau: get_v(tau, non_tau_part), t, ti, limit=20)[0])


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

    log_psi = np.log(psi)
    log_rho = np.log(rho)
    log_rho_n = np.log(rho_n)
    log_sampled = log_psi + log_rho
    log_la = np.log(la)
    log_2la = np.log(2) + log_la
    log_sampled_and_notified = log_sampled + log_rho_n
    log_not_rho_n = np.log(1 - rho_n)
    log_sampled_not_notified = log_sampled + log_not_rho_n
    log_psi_n = np.log(psi_n)
    log_psi_rho = log_psi + log_rho
    log_psi_n_rho_n = log_psi_n + log_rho_n
    log_psi_n_not_rho_n = log_psi_n + log_not_rho_n

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
            # ti = getattr(n, TIME)
            # t = ti - n.dist
            # c3 = get_c3(la, psi, rho, T, ti, c1=c1)
            # # n might have notified a hidden partner who did not get sampled
            # result = [get_log_hidden_partner_notification(t, la, psi, rho, psi_n, T, ti, c1=c1, c3=c3, c4=c4)
            #           + log_sampled_and_notified]
            # or n might have simply not notified
            # result.append(getattr(n, 'p') + log_sampled_not_notified)
            # result = np.array(result, dtype=np.float64)
            # factors = rescale_log(result)
            # n.add_feature('LU', np.log(np.sum(np.exp(result))) - factors)


    # def notification_between_children(child_1, child_2):
    #     # child_1 notified child_2
    #     result = []
    #     if child_1.is_leaf():
    #         tn = getattr(child_1, TIME)
    #         log_ln1 = getattr(child_1, 'LN')
    #         E2 = get_E2(c1=c1, c2=c2, ti=tn, T=T)
    #         for (tip, c) in getattr(child_2, 'CP'):
    #             ti = getattr(tip, TIME)
    #             t = ti - tip.dist
    #             E1 = get_E1(c1=c1, c2=c2, t=t, T=T)
    #             # if we were in time to notify, invoke the notification scenario at tn
    #             if t <= tn <= ti:
    #                 result.append(
    #                     log_ln1 + c + get_log_po(la=la, psi=psi, c1=c1, t=t, ti=tn, E1=E1, E2=E2)
    #                     + psi_n * (tn - ti) + log_psi_n)
    #             # else just invoke the partner (i.e. oriented branch) scenario
    #             elif t <= tn:
    #                 result.append(log_ln1 + c + getattr(tip, 'po') + log_psi_rho)
    #     return result

    def get_lpn_tips(partner, notifier):
        ti = getattr(partner, TIME)
        tr = getattr(notifier, TIME)
        t = ti - partner.dist
        if tr < t:
            return -np.inf
        if tr > ti:
            return getattr(partner, 'LP-late-n')
        else:
            return getattr(partner, 'LP-n') + get_log_pnh(la, psi, t, tr) - psi_n * (ti - tr)

    def get_lpnn_tips(partner, notifier):
        ti = getattr(partner, TIME)
        tr = getattr(notifier, TIME)
        t = ti - partner.dist
        if tr > ti:
            return getattr(partner, 'LP-late-nn')
        else:
            E1 = get_E1(c1=c1, c2=c2, t=t, T=T)
            E2 = get_E2(c1=c1, c2=c2, ti=tr, T=T)
            return getattr(partner, 'LP-nn') + get_log_po(la, psi, c1, t=t, ti=tr, E1=E1, E2=E2) - psi_n * (ti - tr)

    def get_lp_internal(partner, notifier):

        ti = getattr(partner, TIME)
        tr = getattr(notifier, TIME)
        t = ti - partner.dist
        if tr < t:
            return -np.inf

        child_1, child_2 = partner.children
        log_po_la = getattr(n, 'po') + log_la
        is_tip_1 = child_1.is_leaf()
        is_tip_2 = child_2.is_leaf()
        if not is_tip_1 and not is_tip_2:
            result = [
                # the partner is somewhere in child 1's subtree
                get_lp_internal(child_1, notifier) + getattr(child_2, 'LU'),
                # the partner is somewhere in child 2's subtree
                get_lp_internal(child_2, notifier) + getattr(child_1, 'LU')
            ]
            result = np.array(result, dtype=np.float64)
            factors = rescale_log(result)
            return log_po_la + np.log(np.sum(np.exp(result))) - factors
        if is_tip_1 and is_tip_2:
            t1 = getattr(child_1, TIME)
            t2 = getattr(child_2, TIME)
            tr = getattr(notifier, TIME)

            lpn_child1_by_child2 = get_lpn_tips(child_1, child_2)
            lpn_child1_by_notifier = get_lpn_tips(child_1, notifier)
            lpn_child1_by_first_notifier_child2 = lpn_child1_by_notifier if tr <= t2 else lpn_child1_by_child2
            lpn_child2_by_child1 = get_lpn_tips(child_2, child_1)
            lpn_child2_by_notifier = get_lpn_tips(child_2, notifier)
            lpn_child2_by_first_notifier_child1 = lpn_child2_by_notifier if tr <= t1 else lpn_child2_by_child1
            lpnn_child2_by_child1 = get_lpnn_tips(child_2, child_1)
            lpnn_child2_by_notifier = get_lpnn_tips(child_2, notifier)
            lpnn_child2_by_first_notifier_child1 = lpnn_child2_by_notifier if tr <= t1 else lpnn_child2_by_child1
            lpnn_child1_by_notifier = get_lpnn_tips(child_1, notifier)
            lpnn_child1_by_child2 = get_lpnn_tips(child_1, child_2)
            lpnn_child1_by_first_notifier_child2 = lpnn_child1_by_notifier if tr <= t2 else lpnn_child1_by_child2
            result = [
                # child 1 was notified by both child 2 and the notifier
                lpnn_child1_by_first_notifier_child2 + getattr(child_2, 'LU-n'),
                # child 2 was notified by both child 1 and the notifier
                lpnn_child2_by_first_notifier_child1 + getattr(child_1, 'LU-n'),
                # child 1 was notified by both child 2 and the notifier, and also notified child 2
                lpn_child1_by_first_notifier_child2 + lpn_child2_by_child1,
                # child 2 was notified by both child 1 and the notifier, and also notified child 1
                lpn_child2_by_first_notifier_child1 + lpn_child1_by_child2,
                # child 1 was notified only by the notifier, and notified child 2
                lpn_child1_by_notifier + lpnn_child2_by_child1,
                # child 2 was notified only by the notifier, and notified child 1
                lpn_child2_by_notifier + lpnn_child1_by_child2,
                # child 1 was notified only by the notifier, and did not notify child 2
                lpnn_child1_by_notifier + getattr(child_2, 'LU-nn'),
                # child 2 was notified only by the notifier, and did not notify child 1
                lpnn_child2_by_notifier + getattr(child_1, 'LU-nn')
            ]
            result = np.array(result, dtype=np.float64)
            factors = rescale_log(result)
            return log_po_la + np.log(np.sum(np.exp(result))) - factors
        if is_tip_1:
            t1 = getattr(child_1, TIME)
            tr = getattr(notifier, TIME)

            lp_child2_by_child1 = get_lp_internal(child_2, child_1)
            lp_child2_by_notifier = get_lp_internal(child_2, notifier)
            lp_child2_by_first_notifier_child1 = lp_child2_by_notifier if tr <= t1 else lp_child2_by_child1
            result = [
                # child 1 was notified by the notifier and notified child 2
                get_lpn_tips(child_1, notifier) + lp_child2_by_child1,
                # child 1 was notified by the notifier and did not notify child 2
                get_lpnn_tips(child_1, notifier) + getattr(child_2, 'LU'),
                # child 2 was notified by both child 1 and the notifier
                lp_child2_by_first_notifier_child1 + getattr(child_1, 'LU-n'),
                # child 2 was notified only by the notifier
                lp_child2_by_notifier + getattr(child_1, 'LU-nn')
            ]
            result = np.array(result, dtype=np.float64)
            factors = rescale_log(result)
            return log_po_la + np.log(np.sum(np.exp(result))) - factors
        if is_tip_2:
            t2 = getattr(child_1, TIME)
            tr = getattr(notifier, TIME)

            lp_child1_by_child2 = get_lp_internal(child_1, child_2)
            lp_child1_by_notifier = get_lp_internal(child_1, notifier)
            lp_child1_by_first_notifier_child2 = lp_child1_by_notifier if tr <= t2 else lp_child1_by_child2
            result = [
                # child 2 was notified by the notifier and notified child 1
                get_lpn_tips(child_2, notifier) + lp_child1_by_child2,
                # child 2 was notified by the notifier and did not notify child 1
                get_lpnn_tips(child_2, notifier) + getattr(child_1, 'LU'),
                # child 1 was notified by both child 2 and the notifier
                lp_child1_by_first_notifier_child2 + getattr(child_2, 'LU-n'),
                # child 1 was notified only by the notifier
                lp_child1_by_notifier + getattr(child_2, 'LU-nn')
            ]
            result = np.array(result, dtype=np.float64)
            factors = rescale_log(result)
            return log_po_la + np.log(np.sum(np.exp(result))) - factors

    def process_tree(tree):
        for n in tree.traverse('postorder'):
            if n.is_leaf():
                n.add_feature('LU-nn', getattr(n, 'p') + log_sampled_not_notified)
                lun = getattr(n, 'pnh') + log_sampled_and_notified
                n.add_feature('LU-n', lun)
                n.add_feature('LP-late-n', lun)
                n.add_feature('LP-n', log_psi_n_rho_n)
                n.add_feature('LP-late-nn', getattr(n, 'po') + log_sampled_not_notified)
                n.add_feature('LP-nn', log_psi_n_not_rho_n)
                # n.add_feature('CP', [(n, 1)])
                continue

            # Add precalculated probabilities for this node being someone's partner (till one of its tips)
            child_1, child_2 = n.children
            is_tip_1 = child_1.is_leaf()
            is_tip_2 = child_2.is_leaf()

            log_p_2la = getattr(n, 'p') + log_2la
            # lu_1 = getattr(child_1, 'LU')
            # lu_2 = getattr(child_2, 'LU')

            if not is_tip_1 and not is_tip_2:
                n.add_feature('LU', log_p_2la + getattr(child_1, 'LU') + getattr(child_2, 'LU'))
            elif is_tip_1 and is_tip_2:
                result = [
                    # no notification
                    getattr(child_1, 'LU-nn') + getattr(child_2, 'LU-nn'),
                    # child 1 notified child 2
                    getattr(child_1, 'LU-n') + get_lpnn_tips(child_2, child_1),
                    # child 1 notified child 2
                    getattr(child_2, 'LU-n') + get_lpnn_tips(child_1, child_2),
                    # children notified each other
                    get_lpn_tips(child_1, child_2) + get_lpn_tips(child_2, child_1)
                ]
                result = np.array(result, dtype=np.float64)
                factors = rescale_log(result)
                n.add_feature('LU', log_p_2la + np.log(np.sum(np.exp(result))) - factors)
            elif is_tip_1:
                result = [
                    # no notification
                    getattr(child_1, 'LU-nn') + getattr(child_2, 'LU'),
                    # child 1 notified someone in child 2's subtree
                    getattr(child_1, 'LU-n') + get_lp_internal(child_2, child_1)
                ]
                result = np.array(result, dtype=np.float64)
                factors = rescale_log(result)
                n.add_feature('LU', log_p_2la + np.log(np.sum(np.exp(result))) - factors)
            elif is_tip_2:
                result = [
                    # no notification
                    getattr(child_2, 'LU-nn') + getattr(child_1, 'LU'),
                    # child 2 notified someone in child 1's subtree
                    getattr(child_2, 'LU-n') + get_lp_internal(child_1, child_2)
                ]
                result = np.array(result, dtype=np.float64)
                factors = rescale_log(result)
                n.add_feature('LU', log_p_2la + np.log(np.sum(np.exp(result))) - factors)


            # result = []
            # po_log_la = getattr(n, 'po') + log_la
            # po_log_la_lu2 = po_log_la + lu_2
            # po_log_la_lu1 = po_log_la + lu_1
            # for (tip, c) in getattr(child_1, 'CP'):
            #     result.append((tip, po_log_la_lu2 + c))
            # for (tip, c) in getattr(child_2, 'CP'):
            #     result.append((tip, po_log_la_lu1 + c))
            # n.add_feature('CP', result)
            #
            # # if this node is an unnotified individual,
            # # then either its both children are unnotified as well
            # log_lulu = lu_1 + lu_2
            #
            # result = [log_lulu] \
            #          + notification_between_children(child_1, child_2) + notification_between_children(child_2, child_1)
            # if len(result) == 1:
            #     n.add_feature('LU', log_p_2la + result[0])
            # else:
            #     result = np.array(result, dtype=np.float64)
            #     factors = rescale_log(result)
            #     n.add_feature('LU', log_p_2la + np.log(np.sum(np.exp(result))) - factors)
        return getattr(tree, 'LU') if not tree.is_leaf() else getattr(tree, 'LU-nn')

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
