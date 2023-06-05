import numpy as np

from bdpn.parameter_estimator import get_rough_rate_etimates
from bdpn.tree_manager import TIME


def get_c1(la, psi, rho):
    return np.power(np.power((la - psi), 2) + 4 * la * psi * rho, 0.5)


def get_c2(la, psi, rho, T):
    c1 = get_c1(la, psi, rho)
    return (c1 + la - psi) / (c1 - la + psi) * np.exp(-T * c1)


def get_c3(la, psi, rho, T, ti):
    c1 = get_c1(la, psi, rho)
    c2 = get_c2(la, psi, rho, T)
    return c2 * np.exp(c1 * ti) + 1

#
# def get_u_wa(t, la, psi, rho, T):
#     x = t - T
#     c = np.power(-np.power(la, 2) - 4 * la * psi * rho + 2 * la * psi - np.power(psi, 2), 0.5)
#     return 1 / (2 * la) * (c * np.tan(0.5 * (-x * c + 2 * np.arctan((la - psi) / c))) + la + psi)


def get_u(t, la, psi, rho, T):
    c1 = get_c1(la, psi, rho)
    c2 = get_c2(la, psi, rho, T)
    exp_t_c1 = np.exp(t * c1)
    return (la + psi + c1 * (c2 * exp_t_c1 - 1) / (c2 * exp_t_c1 + 1)) / (2 * la)


def get_log_p(t, la, psi, rho, T, ti):
    c1 = get_c1(la, psi, rho)
    c2 = get_c2(la, psi, rho, T)
    c3 = get_c3(la, psi, rho, T, ti)
    # res1 = (np.power(c3, 2) * np.exp(c1 * (t - ti))) / np.power(c2 * np.exp(c1 * t) + 1, 2)
    res2 = (2 * np.log(c3) + (c1 * (t - ti))) - (2 * np.log(c2 * np.exp(c1 * t) + 1))
    # assert(np.log(res1) == res2)
    return res2


def loglikelihood(tree, la, psi, rho, T):
    n = len(tree)
    res = n * np.log(psi * rho) + (n - 1) * np.log(2 * la)
    for n in tree.traverse('preorder'):
        if not n.is_leaf():
            ti = getattr(n, TIME)
            c1, c2 = n.children
            res += get_log_p(ti, la, psi, rho, T, getattr(c1, TIME)) \
                   + get_log_p(ti, la, psi, rho, T, getattr(c2, TIME))
    return res


def get_bounds_start(tree, la, psi, rho):
    bounds = []
    avg_rate, max_rate, min_rate = get_rough_rate_etimates(tree)
    bounds.extend([[min_rate, max_rate]] * (int(la is None) + int(psi is None)))
    bounds.extend([[1e-3, 1 - 1e-3]] * int(rho is None))
    return np.array(bounds, np.float64), [avg_rate, avg_rate, 0.5]
