import os
from multiprocessing.pool import ThreadPool

import numpy as np
import pandas as pd

from bdpn import bd_model
from bdpn.formulas import get_log_p, get_c1, get_c2, get_E2, get_E1, get_log_po, get_log_pn, get_log_po_from_p_pn, \
    get_u, get_log_pp, get_log_no_event
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


def log_subtraction(log_minuend, log_subtrahend):
    """
    Takes logX1 and logX2 as input and returns log(X1 - X2) as output,
    while taking care of potential under/overflow.

    :param log_minuend: logX1 in the formula above
    :param log_subtrahend: logX2 in the formula above
    :return: log of the difference
    """
    result = np.array([log_minuend, log_subtrahend], dtype=np.float64)
    factors = rescale_log(result)
    return np.log(np.sum(np.exp(result) * [1, -1])) - factors

def log_u_p(t, tr, T, la, psi, rho, phi, c1, E1, E2_tr):
    """
    Calculates a log probability of a partner subtree evolving between the time t and T unobserved,
    provided the first partner notification happened at time tr.
    In practice, two possibilities are summed up:
    (1) the hidden partner branch has not yet got sampled between the notification and T;
    (2) the hidden partner branch got removed before the notification (without sampling).


    :param t: start time of the hidden partner subtree evolution
    :param tr: moment of the earliest partner notification
    :param T: end of sampling time
    :param E1: E1 = c2 * exp(c1 * (t - T)), where c2 = (c1 + la - psi) / (c1 - la + psi)
    :param E2_tr: E2_tr = c2 * exp(c1 * (tr - T))
    :return: log of the hidden partner subtree probability (approximation)
    """
    branch_len = tr - t

    # Oriented (partner) branch evolution from t till the moment of notification (tr)
    log_po = get_log_po(la=la, psi=psi, c1=c1, t=t, ti=tr,
                        E1=E1, E2=E2_tr)
    # Hidden partner branch not yet sampled between the notification and T:
    # oriented branch between t and tr, then notified-partner branch evolution with no event
    log_hidden_partner_not_sampled = log_po + get_log_pp(phi, tr, T)

    # Hidden partner branch removed before the notification:
    # divide the oriented branch evolution
    # by no-removal-event-along-the-branch probability (e^{-psi * half_branch_len})
    # and multiply by the removal-event-along-the-branch probability (1 - e^{-psi * half_branch_len})
    # with no sampling, in a log form
    log_hidden_partner_sampled_before = log_po \
                                        + psi * branch_len + np.log(1 - np.exp(-psi * branch_len)) \
                                        + np.log(1 - rho)

    return log_sum([log_hidden_partner_not_sampled, log_hidden_partner_sampled_before])


def log_observed_partner_non_partner_branch(node, tr, T, la, psi, rho, phi, c1, c2):
    """
    Log probability of the case where the node's branch corresponds to a partner
    who transmitted to someone else before notification,
    then the partner stayed unobserved
    (either because they got removed before notification (without sampling),
    or haven't yet got removed after notification (before T)),
    while the someone else (or someone in their subtree) got sampled.
    Therefore, we model a transmission from a hidden partner along the node's branch.

    :param node: the node, who's branch corresponds to the situation described above
    :param tr: notification time
    :param T: time of the end of the sampling period
    :return: log probability of the node's branch + subtree under this case
    """
    ti = getattr(node, TIME)
    tj = ti - node.dist

    # The case where notification happened before the branch start is impossible
    # as partners do not transmit once notified
    if tr < tj:
        return -np.inf

    # Let's assume the partner-to-someone-else transmission happened at the middle of the node's branch
    # (unless the notification happened before)
    observed_partner_branch_len = min((ti - tj) / 2, (tr - tj) * 0.99)
    hidden_transmission_moment = tj + observed_partner_branch_len

    # no removal, but at least one transmission along the oriented branch
    log_observed_partner_branch = log_subtraction(get_log_po(la, psi, c1, t=tj, ti=hidden_transmission_moment,
                                                             E1=get_E1(c1=c1, c2=c2, t=tj, T=T),
                                                             E2=get_E2(c1=c1, c2=c2, ti=hidden_transmission_moment, T=T)),
                                                  get_log_no_event(la + psi, tj, hidden_transmission_moment))
    log_hidden_pb = log_u_p(hidden_transmission_moment, tr, T, la, psi, rho, phi, c1, c2)
    log_someone_else_branch = getattr(node, 'lx' if not node.is_leaf() else 'lxx') - getattr(node, 'logp') \
                              + get_log_p(c1=c1, t=hidden_transmission_moment, ti=ti,
                                          E1=get_E1(c1=c1, c2=c2, t=hidden_transmission_moment, T=T),
                                          E2=get_E2(c1=c1, c2=c2, ti=ti, T=T))
    return log_observed_partner_branch + log_someone_else_branch + log_hidden_pb

def log_mixed_branch(node, tr, T, la, psi, rho, phi, c1, c2, s1, s2):
    """
    Log probability of branch containing a hidden partner.

    :param node: the node, who's branch corresponds to the situation described above
    :param tr: hidden partner notification time
    :param T: time of the end of the sampling period
    :param s1: state for the beginning of the branch (can be 'o' (oriented partner branch) or '-' (standard branch))
    :param s2: state for the end of the branch (can be 'n' (notifier branch) or '-' (standard branch))
    :return: log probability of the node's branch in this case
    """
    ti = getattr(node, TIME)
    tj = ti - node.dist

    # The case where notification happened before the branch start is impossible
    # as partners do not transmit once notified
    if tr < tj:
        return -np.inf

    th = tj + (min(tr, ti) - tj) / 2

    log_no_event = get_log_no_event(la + psi, tj, th)
    E1_tj = get_E1(c1=c1, c2=c2, t=tj, T=T)
    E1_th = get_E1(c1=c1, c2=c2, t=th, T=T)
    E2_th = get_E2(c1=c1, c2=c2, ti=th, T=T)
    E2_ti = get_E2(c1=c1, c2=c2, ti=ti, T=T)
    log_any_events_top = get_log_po(la, psi, c1, t=tj, ti=th, E1=E1_tj, E2=E2_th) if s1 == 'o' \
        else get_log_p(c1, t=tj, ti=th, E1=E1_tj, E2=E2_th)
    log_U = np.log(get_u(la, psi, c1, E1_th))
    log_Up = log_u_p(th, tr, T, la, psi, rho, phi, c1, E1_th, E2_ti)
    log_any_events_bottom = get_log_pn(la, psi, t=th, ti=ti) if s2 == 'n' \
        else get_log_p(c1, t=th, ti=ti, E1=E1_th, E2=E2_ti)
    return log_subtraction(log_any_events_top, log_no_event) + (log_Up - log_U) + log_any_events_bottom


def log_double_mixed_branch(node, tr, T, la, psi, rho, phi, c1, c2):
    """
    Log probability of branch containing two hidden partners (o,-,n).

    :param node: the node, who's branch corresponds to the situation described above
    :param tr: top hidden partner notification time
    :param T: time of the end of the sampling period
    :return: log probability of the node's branch in this case
    """
    ti = getattr(node, TIME)
    tj = ti - node.dist

    # The case where notification happened before the branch start is impossible
    # as partners do not transmit once notified;
    # Also the case where node is not a tip is impossible, as only tips can be notifiers
    if tr < tj or not node.is_leaf():
        return -np.inf

    branch_len_with_hidden_partners = (min(tr, ti) - tj)
    th1 = tj + branch_len_with_hidden_partners / 3
    th2 = tj + 2 / 3 * branch_len_with_hidden_partners

    log_no_event_top = get_log_no_event(la + psi, tj, th1)
    log_no_event_mid = get_log_no_event(la + psi, th1, th2)
    E1_tj = get_E1(c1=c1, c2=c2, t=tj, T=T)
    E1_th1 = get_E1(c1=c1, c2=c2, t=th1, T=T)
    E1_th2 = get_E1(c1=c1, c2=c2, t=th2, T=T)
    E2_th1 = get_E2(c1=c1, c2=c2, ti=th1, T=T)
    E2_th2 = get_E2(c1=c1, c2=c2, ti=th2, T=T)
    E2_tr = get_E2(c1=c1, c2=c2, ti=tr, T=T)
    E2_ti = get_E2(c1=c1, c2=c2, ti=ti, T=T)
    log_any_events_top = get_log_po(la, psi, c1, t=tj, ti=th1, E1=E1_tj, E2=E2_th1)
    log_any_events_mid = get_log_p(c1, t=th1, ti=th2, E1=E1_th1, E2=E2_th2)
    log_any_events_bottom = get_log_pn(la, psi, t=th2, ti=ti)
    log_U_top = np.log(get_u(la, psi, c1, E1_th1))
    log_U_mid = np.log(get_u(la, psi, c1, E1_th2))
    log_Up_top = log_u_p(th1, tr, T, la, psi, rho, phi, c1, E1_th1, E2_tr)
    log_Up_mid = log_u_p(th2, ti, T, la, psi, rho, phi, c1, E1_th2, E2_ti)
    return log_subtraction(log_any_events_top, log_no_event_top) + (log_Up_top - log_U_top) \
        + log_subtraction(log_any_events_mid, log_no_event_mid) + (log_Up_mid - log_U_mid)\
        + log_any_events_bottom


def loglikelihood(forest, la, psi, phi, rho, upsilon, T=None, threads=1):
    annotate_forest_with_time(forest)
    T = get_T(T, forest)

    c1 = get_c1(la=la, psi=psi, rho=rho)
    c2 = get_c2(la=la, psi=psi, c1=c1)

    log_la, log_psi, log_phi, log_rho, log_not_rho, log_ups, log_not_ups, log_2 = \
        np.log(la), np.log(psi), np.log(phi), np.log(rho), np.log(1 - rho), np.log(upsilon), np.log(
            1 - upsilon), np.log(2)
    log_2_la = log_2 + log_la
    log_psi_rho_ups = log_psi + log_rho + log_ups
    log_psi_rho_not_ups = log_psi + log_rho + log_not_ups

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

        node.add_feature('logp', log_p)
        node.add_feature('logpo', log_po)

        if node.is_leaf():
            node.add_feature(
                'lxx',
                log_sum([log_p + log_psi_rho_not_ups,
                         log_mixed_branch(node, ti, T, la, psi, rho, phi, c1, c2, s1='-', s2='n') + log_psi_rho_ups
                         ])
            )
            node.add_feature('lxn', log_pn + log_psi_rho_ups)
            node.add_feature(
                'lnx_late',
                log_sum([log_po + log_psi_rho_not_ups,
                         log_mixed_branch(node, ti, T, la, psi, rho, phi, c1, c2, s1='o', s2='n') + log_psi_rho_ups]))
            node.add_feature('lnn_late', log_pn + log_psi_rho_ups)
            return

        i0, i1 = node.children
        is_tip0, is_tip1 = i0.is_leaf(), i1.is_leaf()

        branch = log_p + log_2_la

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
            return -np.inf, -np.inf

        log_mixed_o_standard = log_mixed_branch(tip, tr, T, la, psi, rho, phi, c1, c2, s1='o', s2='-') \
                               + log_psi_rho_not_ups
        log_mixed_o_standard_n = log_double_mixed_branch(tip, tr, T, la, psi, rho, phi, c1, c2) \
                                 + log_psi_rho_ups
        if tr > ti:
            return log_sum([getattr(tip, 'lnx_late'),
                            log_mixed_o_standard,
                            log_mixed_o_standard_n]), \
                getattr(tip, 'lnn_late')
        else:
            log_po = get_log_po(la=la, psi=psi, c1=c1, t=tj, ti=tr, E1=get_E1(c1=c1, c2=c2, t=tj, T=T),
                                E2=get_E2(c1=c1, c2=c2, ti=tr, T=T))
            log_pn = get_log_pn(la=la, psi=psi, t=tj, ti=tr)
            log_pp = get_log_pp(phi=phi, t=tr, ti=ti)
            return log_sum([log_po + log_pp + log_phi + log_not_ups,
                            log_mixed_branch(tip, tr, T, la, psi, rho, phi, c1, c2, s1='o', s2='n') + log_psi_rho_ups,
                            log_mixed_o_standard,
                            log_mixed_o_standard_n]), \
                log_pn + log_pp + log_phi + log_ups

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

        observed_branch = getattr(node, 'logpo') + log_la
        mixed_branch = log_mixed_branch(node, tr, T, la, psi, rho, phi, c1, c2, 'o', '-') + log_2_la

        i0, i1 = node.children
        is_tip0, is_tip1 = i0.is_leaf(), i1.is_leaf()

        if not is_tip0 and not is_tip1:
            lx0, lx1 = getattr(i0, 'lx'), getattr(i1, 'lx')
            return log_sum([observed_branch + log_sum([get_log_ln_internal(i0, notifier) + lx1,
                                                       lx0 + get_log_ln_internal(i1, notifier)]),
                            mixed_branch + lx0 + lx1])

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

            log_lxn_0,  log_lxn_1 = getattr(i0, 'lxn'), getattr(i1, 'lxn')
            log_lxx_0,  log_lxx_1 = getattr(i0, 'lxx'), getattr(i1, 'lxx')
            return log_sum([observed_branch + log_sum([log_lnx_0_by_both + log_lxn_1,
                                                       log_lnn_0_by_both + log_lnn_1_by_0,
                                                       log_lxn_0 + log_lnx_1_by_both,
                                                       log_lnn_0_by_1 + log_lnn_1_by_both,
                                                       log_lnn_0_by_r + log_lnx_1_by_0,
                                                       log_lnx_0_by_1 + log_lnn_1_by_r,
                                                       log_lnx_0_by_r + log_lxx_1,
                                                       log_lxx_0 + log_lnx_1_by_r]),
                            mixed_branch + log_sum([log_lnx_0_by_1 + log_lxn_1,
                                                    log_lnn_0_by_1 + log_lnn_1_by_0,
                                                    log_lxn_0 + log_lnx_1_by_0,
                                                    log_lxx_0 + log_lxx_1])])

        # i0 is a tip and i1 is internal
        if is_tip1:
            i0, i1 = i1, i0
            ti0, ti1 = ti1, ti0
        log_lnx_0_by_r, log_lnn_0_by_r = get_log_lnx_lnn_tip(i0, notifier)
        log_lx_1 = getattr(i1, 'lx')
        log_lxn_0 = getattr(i0, 'lxn')
        log_lxx_0 = getattr(i0, 'lxn')
        log_ln_1_by_0 = get_log_ln_internal(i1, i0)
        log_ln_1_by_r = get_log_ln_internal(i1, notifier)
        log_ln_1_by_both = log_ln_1_by_0 if ti0 < tr else log_ln_1_by_r
        return log_sum([observed_branch + log_sum([log_lnn_0_by_r + log_ln_1_by_0,
                                                   log_lnx_0_by_r + log_lx_1,
                                                   log_lxn_0 + log_ln_1_by_both,
                                                   log_lxx_0 + log_ln_1_by_r]),
                        mixed_branch + log_sum([log_lxn_0 + log_ln_1_by_0,
                                                log_lxx_0 + log_lx_1])])

    def process_tree(tree):
        for node in tree.traverse('postorder'):
            process_node(node)
        return getattr(tree, 'lx' if not tree.is_leaf() else 'lxx')

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
    #     params.psi = psi

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
    print('Fixed input parameter(s): {}'
          .format(', '.join('{}={:g}'.format(*_)
                            for _ in zip(PARAMETER_NAMES[input_params != None], input_params[input_params != None]))))
    print('Starting BDPN parameters: {}'.format(start_parameters))
    vs, cis, lk = optimize_likelihood_params(forest, input_parameters=input_params,
                                             loglikelihood=loglikelihood, bounds=bounds[input_params == None],
                                             start_parameters=start_parameters, cis=ci)
    print('Estimated BDPN parameters: {}'.format(vs))
    if ci:
        print('Estimated CIs:\n{}'.format(cis))
    return vs, cis


if '__main__' == __name__:
    main()
