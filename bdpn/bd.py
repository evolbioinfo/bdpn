import numpy as np

from bdpn.tree_manager import TIME, annotate_tree
from bdpn.formulas import get_c1, get_c2, get_E2, get_E1, get_log_p, get_u


def loglikelihood(forest, la, psi, rho, T=None, threads=1):
    for tree in forest:
        if not hasattr(tree, TIME):
            annotate_tree(tree)
    if T is None:
        T = 0
        for tree in forest:
            T = max(T, max(getattr(_, TIME) for _ in tree))

    c1 = get_c1(la=la, psi=psi, rho=rho)
    c2 = get_c2(la=la, psi=psi, c1=c1)

    log_psi_rho = np.log(psi) + np.log(rho)
    log_two_la = np.log(2) + np.log(la)

    res = 0
    for tree in forest:
        n = len(tree)
        res += n * log_psi_rho + (n - 1) * log_two_la
        for n in tree.traverse('preorder'):
            if not n.is_leaf():
                t = getattr(n, TIME)
                E1 = get_E1(c1=c1, c2=c2, t=t, T=T)
                child1, child2 = n.children
                ti_1 = getattr(child1, TIME)
                ti_2 = getattr(child2, TIME)
                res += get_log_p(c1, t, ti=ti_1, E1=E1, E2=get_E2(c1=c1, c2=c2, ti=ti_1, T=T)) \
                       + get_log_p(c1, t, ti=ti_2, E1=E1, E2=get_E2(c1=c1, c2=c2, ti=ti_2, T=T))
    u = get_u(la, psi, c1, E1=get_E1(c1=c1, c2=c2, t=0, T=T))
    return res + len(forest) * u / (1 - u) * np.log(u)

