import os
from multiprocessing.pool import ThreadPool

import numpy as np

from bdpn import bd_model
from bdpn.formulas import get_log_p, get_c1, get_c2, get_E, get_log_ppb, get_log_pn, get_log_ppb_from_p_pn, \
    get_u, get_log_no_event, get_log_ppa, get_log_ppa_from_ppb, get_log_pb
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
    # print(log_subtrahend, log_minuend)
    # assert (log_subtrahend < log_minuend)
    result = np.array([log_minuend, log_subtrahend], dtype=np.float64)
    factors = rescale_log(result)
    return np.log(np.sum(np.exp(result) * [1, -1])) - factors


def log_u_p(t, tr, T, la, psi, rho, phi, c1, c2, E_t, E_tr):
    """
    Calculates a log probability of a partner subtree evolving between the time t and T unobserved,
    provided the first partner notification happened at time tr.
    In practice, two possibilities are summed up:
    (1) the hidden partner branch has not yet got sampled between the notification and T;
    (2) the hidden partner branch got removed before the notification (without sampling).


    :param t: start time of the hidden partner subtree evolution
    :param tr: moment of the earliest partner notification
    :param T: end of sampling time
    :param E_t: E_t = c2 * exp(c1 * (t - T)), where c2 = (c1 + la - psi) / (c1 - la + psi)
    :param E_tr: E_tr = c2 * exp(c1 * (tr - T))
    :return: log of the hidden partner subtree probability (approximation)
    """
    E_T = get_E(c1, c2, T, T)
    if tr > t:
        branch_len = tr - t

        # Partner branch evolution from t till the moment of notification (tr)
        log_ppb = get_log_ppb(la=la, psi=psi, c1=c1, t=t, ti=tr, E_t=E_t, E_ti=E_tr)
        # Notified-partner branch evolution after tr
        log_ppa = get_log_ppa(la, psi, phi, c1, tr, T, E_t=E_tr, E_ti=E_T)
        log_hidden_partner_not_sampled = log_ppb + log_ppa

        # Hidden partner branch removed before the notification:
        # divide the oriented branch evolution
        # by no-removal-event-along-the-branch probability (e^{-psi * branch_len})
        # and multiply by the removal-event-along-the-branch probability (1 - e^{-psi * branch_len})
        # with no sampling, in a log form
        log_hidden_partner_sampled_before = log_ppb \
                                            + psi * branch_len + np.log(1 - np.exp(-psi * branch_len)) \
                                            + np.log(1 - rho)

        return log_sum([log_hidden_partner_not_sampled, log_hidden_partner_sampled_before])
    else:
        return get_log_ppa(la, psi, phi, c1, t, T, E_t=E_t, E_ti=E_T)


def get_log_pp(la, psi, phi, c1, c2, tr, T, node=None, tj=None, ti=None):
    if ti is None:
        ti = getattr(node, TIME)
    if tj is None:
        tj = ti - node.dist

    E_tj = get_E(c1=c1, c2=c2, t=tj, T=T)
    E_ti = get_E(c1, c2, ti, T)
    if tr < tj:
        return getattr(node, 'log_ppa') if node is not None \
            else get_log_ppa(la, psi, phi, c1, t=tj, ti=ti, E_t=E_tj, E_ti=E_ti)
    if tr > ti:
        return getattr(node, 'log_ppb') if node is not None \
            else get_log_ppb(la, psi, c1, t=tj, ti=ti, E_t=E_tj, E_ti=E_ti)

    E_tr = get_E(c1, c2, tr, T)
    return get_log_ppb(la, psi, c1, t=tj, ti=tr, E_t=E_tj, E_ti=E_tr) \
        + get_log_ppa(la, psi, phi, c1, t=tr, ti=ti, E_t=E_tr, E_ti=E_ti)


def log_mixed_branch(node, tr, T, la, psi, rho, phi, c1, c2, s1, s2):
    """
    Log probability of branch containing a hidden partner.

    :param node: the node, who's branch corresponds to the situation described above
    :param tr: hidden partner notification time
    :param T: time of the end of the sampling period
    :param s1: state for the beginning of the branch (can be 'o' (oriented partner branch) or '-' (standard branch) or 'p' (notified partner))
    :param s2: state for the end of the branch (can be 'n' (notifier branch) or '-' (standard branch))
    :return: log probability of the node's branch in this case
    """
    ti = getattr(node, TIME)
    tj = ti - node.dist
    if s1 == 'p':
        if s2 == 'n':
            if tr < tj:
                return getattr(node, 'log_pa-n')
            if tr > ti:
                return getattr(node, 'log_pb-n')
        elif s2 == '-' and tr < tj:
            return getattr(node, 'log_pa--')
    elif s1 == '-' and s2 == 'n':
        return getattr(node, 'log_--n')

    th = tj + (ti - tj) / 2

    def get_mixed_top(node):
        E_tj = get_E(c1=c1, c2=c2, t=tj, T=T)
        E_th = get_E(c1, c2, th, T)
        E_tr = get_E(c1, c2, tr, T)
        return get_log_ppb(la, psi, c1, t=tj, ti=tr, E_t=E_tj, E_ti=E_tr) \
            + log_subtraction(get_log_ppa(la, psi, phi, c1, t=tr, ti=th, E_t=E_tr, E_ti=E_th),
                              get_log_no_event(la + phi, tr, th)) \
            - getattr(node, 'log_u_th')

    log_one_transmission_top = getattr(node, 'log_top_-') if s1 == '-' \
        else (getattr(node, 'log_top_pa') if tr < tj
              else (getattr(node, 'log_top_pb') if tr > th else get_mixed_top(node)))
    log_Up = log_u_p(th, tr, T, la, psi, rho, phi, c1, c2,
                     E_t=get_E(c1=c1, c2=c2, t=th, T=T), E_tr=get_E(c1, c2, tr, T))
    log_any_events_bottom = getattr(node, 'log_bottom_n' if s2 == 'n' else 'log_bottom_-')
    return log_one_transmission_top + log_Up + log_any_events_bottom


def log_double_mixed_branch(node, tr, T, la, psi, rho, phi, c1, c2):
    """
    Log probability of branch containing two hidden partners (p,-,n).

    :param node: the node, who's branch corresponds to the situation described above
    :param tr: top hidden partner notification time
    :param T: time of the end of the sampling period
    :return: log probability of the node's branch in this case
    """
    ti = getattr(node, TIME)
    tj = ti - node.dist

    th1 = tj + (ti - tj) / 3

    if tr < tj:
        return getattr(node, 'log_pa---n')

    def get_mixed_top(node):
        E_tj = get_E(c1=c1, c2=c2, t=tj, T=T)
        E_th1 = get_E(c1, c2, th1, T)
        E_tr = get_E(c1, c2, tr, T)
        return get_log_ppb(la, psi, c1, t=tj, ti=tr, E_t=E_tj, E_ti=E_tr) \
            + log_subtraction(get_log_ppa(la, psi, phi, c1, t=tr, ti=th1, E_t=E_tr, E_ti=E_th1),
                              get_log_no_event(la + phi, tr, th1)) \
            - getattr(node, 'log_u_th1')

    log_one_transmission_top = getattr(node, 'log_top1_pa') if tr < tj \
        else (getattr(node, 'log_top1_pb') if tr > th1 else get_mixed_top(node))
    log_Up = log_u_p(th1, tr, T, la, psi, rho, phi, c1, c2,
                     E_t=get_E(c1=c1, c2=c2, t=th1, T=T), E_tr=get_E(c1, c2, tr, T))
    return log_one_transmission_top + log_Up + getattr(node, 'log_bottom_--n')


def loglikelihood(forest, la, psi, phi, rho, upsilon, T=None, threads=1):
    annotate_forest_with_time(forest)
    T = get_T(T, forest)

    c1 = get_c1(la=la, psi=psi, rho=rho)
    c2 = get_c2(la=la, psi=psi, c1=c1)

    log_la, log_psi, log_phi, log_rho, log_not_rho, log_ups, log_not_ups, log_2 = \
        np.log(la), np.log(psi), np.log(phi), np.log(rho), np.log(1 - rho), np.log(upsilon), \
            np.log(1 - upsilon), np.log(2)
    log_2_la = log_2 + log_la
    log_psi_rho_ups = log_psi + log_rho + log_ups
    log_psi_rho_not_ups = log_psi + log_rho + log_not_ups
    log_phi_ups = log_phi + log_ups
    log_phi_not_ups = log_phi + log_not_ups

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

        E_tj = get_E(c1=c1, c2=c2, t=tj, T=T)
        E_ti = get_E(c1, c2, ti, T)

        log_p = get_log_p(c1=c1, t=tj, ti=ti, E_t=E_tj, E_ti=E_ti)
        log_pn = get_log_pn(la=la, psi=psi, t=tj, ti=ti)
        log_ppb = get_log_ppb_from_p_pn(log_p=log_p, log_pn=log_pn)
        log_ppa = get_log_ppa_from_ppb(log_ppb=log_ppb, psi=psi, phi=phi, t=tj, ti=ti)

        node.add_feature('log_ppb', log_ppb)
        node.add_feature('log_ppa', log_ppa)

        th = tj + (ti - tj) / 2
        E_th = get_E(c1, c2, th, T)
        E_T = get_E(c1, c2, T, T)
        log_u_th = np.log(get_u(la, psi, c1, E_th))
        log_no_event_th = get_log_no_event(la + psi, tj, th)
        node.add_feature('log_u_th', log_u_th)
        log_top_ppb = log_subtraction(get_log_ppb(la, psi, c1, tj, th, E_tj, E_th), log_no_event_th) - log_u_th
        node.add_feature('log_top_pb', log_top_ppb)
        log_top_pa = log_subtraction(get_log_ppa(la, psi, phi, c1, tj, th, E_tj, E_th),
                                     get_log_no_event(la + phi, tj, th)) - log_u_th
        node.add_feature('log_top_pa', log_top_pa)
        log_top_pa_hidden_partner = \
            getattr(node, 'log_top_pa') + get_log_ppa(la, psi, phi, c1, th, T, E_th, E_T)
        log_top_ = log_subtraction(get_log_p(c1, tj, th, E_tj, E_th), log_no_event_th) - log_u_th
        node.add_feature('log_top_-', log_top_)
        log_bottom_ = get_log_p(c1, th, ti, E_th, E_ti)
        node.add_feature('log_bottom_-', log_bottom_)
        node.add_feature('log_pa--', log_top_pa_hidden_partner + log_bottom_)

        if node.is_leaf():
            log_bottom_n = get_log_pn(la, psi, th, ti)
            node.add_feature('log_bottom_n', log_bottom_n)
            node.add_feature('log_pa-n', log_top_pa_hidden_partner + log_bottom_n)
            log_u_p_ti_bottom_n = log_u_p(th, ti, T, la, psi, rho, phi, c1, c2, E_th, E_ti) + log_bottom_n
            node.add_feature('log_pb-n', log_top_ppb + log_u_p_ti_bottom_n)
            node.add_feature('log_--n', log_top_ + log_u_p_ti_bottom_n)
            th1 = tj + (ti - tj) / 3
            th2 = tj + 2 * (ti - tj) / 3
            E_th1 = get_E(c1, c2, th1, T)
            E_th2 = get_E(c1, c2, th2, T)

            log_u_th1 = np.log(get_u(la, psi, c1, E_th1))
            log_u_th2 = np.log(get_u(la, psi, c1, E_th2))
            node.add_feature('log_u_th1', log_u_th1)
            log_top1_pb = log_subtraction(get_log_ppb(la, psi, c1, tj, th1, E_tj, E_th1),
                                          get_log_no_event(la + psi, tj, th1)) - log_u_th1
            node.add_feature('log_top1_pb', log_top1_pb)
            log_top1_pa = log_subtraction(get_log_ppa(la, psi, phi, c1, tj, th1, E_tj, E_th1),
                                          get_log_no_event(la + phi, tj, th1)) - log_u_th1
            node.add_feature('log_top1_pa', log_top1_pa)
            log_bottom__n = log_subtraction(get_log_p(c1, th1, th2, E_th1, E_th2),
                                            get_log_no_event(la + psi, th1, th2)) - log_u_th2 \
                            + log_u_p(th2, ti, T, la, psi, rho, phi, c1, c2, E_th2, E_ti) \
                            + get_log_pn(la, psi, th2, ti)
            node.add_feature('log_bottom_--n', log_bottom__n)
            log_pa__n = log_top1_pa + get_log_ppa(la, psi, phi, c1, th, T, E_th1, E_T) + log_bottom__n
            node.add_feature('log_pa---n', log_pa__n)

            node.add_feature('lxx', log_sum([log_p + log_psi_rho_not_ups,
                                             getattr(node, 'log_--n') + log_psi_rho_ups]))
            node.add_feature('lxn', log_pn + log_psi_rho_ups)
            node.add_feature('lnx_early', log_sum([log_ppa + log_phi_not_ups,
                                                   getattr(node, 'log_pa--') + log_psi_rho_not_ups,
                                                   getattr(node, 'log_pa-n') + log_psi_rho_ups,
                                                   getattr(node, 'log_pa---n') + log_psi_rho_ups]))
            node.add_feature('lnx_late', log_sum([log_ppb + log_psi_rho_not_ups,
                                                  getattr(node, 'log_pb-n') + log_psi_rho_ups]))
            node.add_feature('lnn_early', get_log_pb(la=la, phi=phi, t=tj, ti=ti) + log_phi_ups)
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
            return getattr(tip, 'lnx_early'), getattr(tip, 'lnn_early')
        if tr > ti:
            return log_sum([getattr(tip, 'lnx_late'),
                            log_mixed_branch(tip, tr, T, la, psi, rho, phi, c1, c2, s1='p', s2='-')
                            + log_psi_rho_not_ups,
                            log_double_mixed_branch(tip, tr, T, la, psi, rho, phi, c1, c2) + log_psi_rho_ups]), \
                getattr(tip, 'lnn_late')

        return log_sum([get_log_pp(la, psi, phi, c1, c2, tr, T, tip) + log_phi_not_ups,
                        log_mixed_branch(tip, tr, T, la, psi, rho, phi, c1, c2, s1='p', s2='-') + log_psi_rho_not_ups,
                        log_mixed_branch(tip, tr, T, la, psi, rho, phi, c1, c2, s1='p', s2='n') + log_psi_rho_ups,
                        log_double_mixed_branch(tip, tr, T, la, psi, rho, phi, c1, c2) + log_psi_rho_ups]), \
            get_log_pn(la=la, psi=psi, t=tj, ti=tr) + get_log_pb(la=la, phi=phi, t=tr, ti=ti) + log_phi_ups

    def get_log_ln_internal(node, notifier):
        """
        Calculates loglikelihood density of a partner node branch.

        :param node: partner node
        :param notifier: tip node corresponding to the node's notifier
        :return: loglikelihood density
        """
        tr = getattr(notifier, TIME)

        observed_branch = get_log_pp(la, psi, phi, c1, c2, tr, T, node=node) + log_la
        mixed_branch = log_mixed_branch(node, tr, T, la, psi, rho, phi, c1, c2, 'p', '-') + log_2_la

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

            log_lxn_0, log_lxn_1 = getattr(i0, 'lxn'), getattr(i1, 'lxn')
            log_lxx_0, log_lxx_1 = getattr(i0, 'lxx'), getattr(i1, 'lxx')
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

    u = get_u(la, psi, c1, E_t=get_E(c1=c1, c2=c2, t=0, T=T))
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


def loglikelihood_main():
    """
    Entry point for tree likelihood estimation with the BDPN model with command-line arguments.
    :return: void
    """
    import argparse

    parser = \
        argparse.ArgumentParser(description="Calculate BDPN likelihood on a given forest for given parameter values.")
    parser.add_argument('--la', required=True, type=float, help="transmission rate")
    parser.add_argument('--psi', required=True, type=float, help="removal rate")
    parser.add_argument('--rho', required=True, type=float, help='sampling probability')
    parser.add_argument('--upsilon', required=True, type=float, help='notification probability')
    parser.add_argument('--phi', required=True, type=float, help='partner removal rate')
    parser.add_argument('--nwk', required=True, type=str, help="input tree file")
    params = parser.parse_args()

    forest = read_forest(params.nwk)
    lk = loglikelihood(forest, la=params.la, psi=params.psi, rho=params.rho, phi=params.phi, upsilon=params.upsilon)
    print(lk)


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
