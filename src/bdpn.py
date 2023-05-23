import numpy as np
import pandas as pd
from ete3 import Tree
from scipy.optimize import NonlinearConstraint, LinearConstraint, minimize


def annotate_tree(tree):
    for n in tree.traverse('preorder'):
        if n.is_root():
            p_time = 0
        else:
            p_time = getattr(n.up, 'time')
        n.add_feature('time', p_time + n.dist)


def get_c1(la, psi, rho):
    return np.power(np.power((la - psi), 2) + 4 * la * psi * rho, 0.5)


def get_c2(la, psi, rho, T):
    c1 = get_c1(la, psi, rho)
    return (c1 + la - psi) / (c1 - la + psi) * np.exp(-T * c1)


def get_c3(la, psi, rho, T, ti):
    c1 = get_c1(la, psi, rho)
    c2 = get_c2(la, psi, rho, T)
    return c2 * np.exp(c1 * ti) + 1


def get_u(t, la, psi, rho, T):
    c1 = get_c1(la, psi, rho)
    c2 = get_c2(la, psi, rho, T)
    exp_t_c1 = np.exp(t * c1)
    return (la + psi + c1 * (c2 * exp_t_c1 - 1) / (c2 * exp_t_c1 + 1)) / (2 * la)


def get_p(t, la, psi, rho, T, ti):
    c1 = get_c1(la, psi, rho)
    c2 = get_c2(la, psi, rho, T)
    c3 = get_c3(la, psi, rho, T, ti)
    return (np.power(c3, 2) * np.exp(c1 * (t - ti))) / np.power(c2 * np.exp(c1 * t) + 1, 2)


def loglikelihood(tree, la, psi, rho, T):
    n = len(tree)
    res = n * np.log(psi * rho) + (n - 1) * np.log(2 * la)
    for n in tree.traverse('preorder'):
        if not n.is_leaf():
            ti = getattr(n, 'time')
            c1, c2 = n.children
            res += np.log(get_p(ti, la, psi, rho, T, getattr(c1, 'time'))) \
                   + np.log(get_p(ti, la, psi, rho, T, getattr(c2, 'time')))
    return res


def optimize_likelihood_params(tree, T, la=None, psi=None, rho=None):
    """
    Optimizes the likelihood parameters for a given forest and a given MTBD model.


    :param forest: a list of ete3.Tree trees, annotated with node states and times via features STATE_K and TI.
    :param T: time at end of the sampling period
    :param model: MTBD model containing starting parameter values
    :param optimise: MTBD model whose rates indicate which parameters need to optimized:
        positive rates correspond to optimized parameters
    :param u: number of hidden trees, where no tip got sampled
    :return: the values of optimised parameters and the corresponding loglikelihood: (MU, LA, PSI, RHO, best_log_lh)
    """

    bounds = []

    l = []
    for n in tree.traverse('preorder'):
        if n.dist:
            l.append(n.dist)
    max_rate = 10 / np.mean(l)
    min_rate = 1 / np.max(l)
    print('Considering ', max_rate, ' as max rate and ', min_rate, ' as min rate')
    avg_rate = (min_rate + max_rate) / 2

    bounds.extend([[min_rate, max_rate]] * (int(la is None) + int(psi is None)))
    bounds.extend([[1e-3, 1 - 1e-3]] * int(rho is None))
    bounds = np.array(bounds, np.float64)

    def get_real_params_from_optimised(ps):
        ps = np.maximum(np.minimum(ps, bounds[:, 1]), bounds[:, 0])
        result = np.zeros(3)
        i = 0
        if la is None:
            result[0] = ps[i]
            i += 1
        else:
            result[0] = la
        if psi is None:
            result[1] = ps[i]
            i += 1
        else:
            result[1] = psi
        if rho is None:
            result[2] = ps[i]
            i += 1
        else:
            result[2] = rho
        return result

    def get_optimised_params_from_real(ps):
        result = []
        if la is None:
            result.append(ps[0])
        if psi is None:
            result.append(ps[1])
        if rho is None:
            result.append(ps[2])
        return np.array(result)


    def get_v(ps):
        if np.any(np.isnan(ps)):
            return np.nan
        ps = get_real_params_from_optimised(ps)
        res = loglikelihood(tree, *ps, T)
        # print(ps, "\t-->\t", res)
        return -res

    x0 = get_optimised_params_from_real([avg_rate, avg_rate, avg_rate])
    best_log_lh = -get_v(x0)

    def R0(vs):
        vs = get_real_params_from_optimised(vs)
        return vs[0] / vs[1]

    cons = (NonlinearConstraint(R0, 0.2, 100), LinearConstraint(np.eye(len(x0)), bounds[:, 0], bounds[:, 1]),)

    for i in range(10):
        if i == 0:
            vs = x0
        else:
            keep_searching = True
            while keep_searching:
                keep_searching = False
                vs = np.random.uniform(bounds[:, 0], bounds[:, 1])
                for c in cons:
                    if not isinstance(c, LinearConstraint):
                        val = c.fun(vs)
                        if c.lb > val or c.ub < val:
                            keep_searching = True
                            break

        fres = minimize(get_v, x0=vs, method='COBYLA', bounds=bounds, constraints=cons)
        if fres.success and not np.any(np.isnan(fres.x)):
            if -fres.fun >= best_log_lh:
                x0 = np.array(fres.x)
                best_log_lh = -fres.fun
                break
        print('Attempt {} of trying to optimise the parameters: {}.'.format(i, -fres.fun))
    return get_real_params_from_optimised(x0)


nwk = '/home/azhukova/projects/bdpn/trees/bd/tree.1.nwk'
tree = Tree(nwk)
annotate_tree(tree)
df = pd.read_csv(nwk.replace('.nwk', '.log'))
rho = df.loc[0, 'sampling probability']
R0 = df.loc[0, 'R0']
it = df.loc[0, 'infectious time']
psi = 1 / it
la = R0 * psi
T = max(getattr(_, 'time') for _ in tree)
vs = optimize_likelihood_params(tree, T, la=None, psi=None, rho=rho)
print('Real params: {}'.format([la, psi, rho]))
print('Found params: {}'.format(vs))