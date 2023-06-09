import numpy as np

from bdpn.tree_manager import TIME, annotate_tree


def get_c1(la, psi, rho):
    return np.power(np.power((la - psi), 2) + 4 * la * psi * rho, 0.5)


def get_c2(la, psi, rho, T, c1=None):
    if c1 is None:
        c1 = get_c1(la, psi, rho)
    return (c1 + la - psi) / (c1 - la + psi) * np.exp(-T * c1)


def get_c3(la, psi, rho, T, ti, c1=None):
    if c1 is None:
        c1 = get_c1(la, psi, rho)
    # c2 = get_c2(la, psi, rho, T)
    # return c2 * np.exp(c1 * ti) + 1
    # let's inline c2 here to reduce overflow
    return (c1 + la - psi) / (c1 - la + psi) * np.exp(c1 * (ti - T)) + 1

#
# def get_u_wa(t, la, psi, rho, T):
#     x = t - T
#     c = np.power(-np.power(la, 2) - 4 * la * psi * rho + 2 * la * psi - np.power(psi, 2), 0.5)
#     return 1 / (2 * la) * (c * np.tan(0.5 * (-x * c + 2 * np.arctan((la - psi) / c))) + la + psi)


def get_u(t, la, psi, rho, T):
    c1 = get_c1(la, psi, rho)
    # c2 = get_c2(la, psi, rho, T, c1=c1)
    # Inline c2 to avoid overflow
    c2_exp_c1 = (c1 + la - psi) / (c1 - la + psi) * np.exp(c1 * (t - T))
    # return (la + psi + c1 * ((c2_exp_c1 - 1) / (c2_exp_c1 + 1))) / (2 * la)
    return (la + psi) / (2 * la) + c1 / (2 * la) * ((c2_exp_c1 - 1) / (c2_exp_c1 + 1))


def get_log_p(t, la, psi, rho, T, ti):
    c1 = get_c1(la, psi, rho)
    c3 = get_c3(la, psi, rho, T, ti, c1=c1)
    # c2 = get_c2(la, psi, rho, T, c1=c1)
    # return (2 * np.log(c3) + (c1 * (t - ti))) - 2 * np.log(c2 * np.exp(c1 * t) + 1)
    # Inline c2 to avoid overflow
    return 2 * np.log(c3) + (c1 * (t - ti)) - 2 * np.log((c1 + la - psi) / (c1 - la + psi) * np.exp(c1 * (t - T)) + 1)


def loglikelihood(forest, la, psi, rho, T=None):
    for tree in forest:
        if not hasattr(tree, TIME):
            annotate_tree(tree)
    if T is None:
        T = 0
        for tree in forest:
            T = max(T, max(getattr(_, TIME) for _ in tree))

    res = 0
    for tree in forest:
        n = len(tree)
        res += n * np.log(psi * rho) + (n - 1) * np.log(2 * la)
        for n in tree.traverse('preorder'):
            if not n.is_leaf():
                ti = getattr(n, TIME)
                c1, c2 = n.children
                res += get_log_p(ti, la, psi, rho, T, getattr(c1, TIME)) \
                       + get_log_p(ti, la, psi, rho, T, getattr(c2, TIME))
    u = get_u(0, la, psi, rho, T)
    return res + len(forest) * u / (1 - u) * np.log(u)

