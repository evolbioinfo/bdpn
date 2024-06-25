import os

import numpy as np

from bdpn.formulas import get_c1, get_c2, get_E, get_log_p, get_u
from bdpn.parameter_estimator import optimize_likelihood_params
from bdpn.tree_manager import TIME, read_forest, annotate_forest_with_time, get_T

DEFAULT_MIN_PROB = 1e-6
DEFAULT_MAX_PROB = 1
DEFAULT_MIN_RATE = 1e-3
DEFAULT_MAX_RATE = 1e3

DEFAULT_LOWER_BOUNDS = [DEFAULT_MIN_RATE, DEFAULT_MIN_RATE, DEFAULT_MIN_PROB]
DEFAULT_UPPER_BOUNDS = [DEFAULT_MAX_RATE, DEFAULT_MAX_RATE, DEFAULT_MAX_PROB]

PARAMETER_NAMES = np.array(['la', 'psi', 'rho'])


def get_start_parameters(forest, la=None, psi=None, rho=None):
    la_is_fixed = la is not None and la > 0
    psi_is_fixed = psi is not None and psi > 0
    rho_is_fixed = rho is not None and 0 < rho <= 1

    rho_est = rho if rho_is_fixed else 0.5

    if la_is_fixed and psi_is_fixed:
        return np.array([la, psi, rho_est], dtype=np.float64)

    # Let's estimate transmission time as a median internal branch length
    # and sampling time as a median external branch length
    internal_dists, external_dists = [], []
    for tree in forest:
        for n in tree.traverse():
            if n.is_root() and not n.dist:
                continue
            (internal_dists if not n.is_leaf() else external_dists).append(n.dist)

    psi_est = psi if psi_is_fixed else 1 / np.median(external_dists)
    # if it is a corner case when we only have tips, let's use sampling times
    la_est = la if la_is_fixed else ((1 / np.median(internal_dists)) if internal_dists else 1.1 * psi_est)
    if la_est <= psi_est:
        if la_is_fixed:
            psi_est = la_est * 0.9
        else:
            la_est *= psi_est * 1.1

    return np.array([la_est, psi_est, rho_est], dtype=np.float64)


def loglikelihood(forest, la, psi, rho, T, threads=1):
    c1 = get_c1(la=la, psi=psi, rho=rho)
    c2 = get_c2(la=la, psi=psi, c1=c1)

    log_psi_rho = np.log(psi) + np.log(rho)
    log_two_la = np.log(2) + np.log(la)

    u = get_u(la, psi, c1, E_t=get_E(c1=c1, c2=c2, t=0, T=T))
    res = len(forest) * u / (1 - u) * np.log(u)
    for tree in forest:
        n = len(tree)
        res += n * log_psi_rho + (n - 1) * log_two_la
        for n in tree.traverse('preorder'):
            if not n.is_leaf():
                t = getattr(n, TIME)
                E_t = get_E(c1=c1, c2=c2, t=t, T=T)
                child1, child2 = n.children
                ti_1 = getattr(child1, TIME)
                ti_2 = getattr(child2, TIME)
                res += get_log_p(c1, t, ti=ti_1, E_t=E_t, E_ti=get_E(c1, c2, ti_1, T)) \
                       + get_log_p(c1, t, ti=ti_2, E_t=E_t, E_ti=get_E(c1, c2, ti_2, T))
        root_ti = getattr(tree, TIME)
        root_t = root_ti - tree.dist
        res += get_log_p(c1, root_t, ti=root_ti, E_t=get_E(c1, c2, root_t, T), E_ti=get_E(c1, c2, root_ti, T))
    return res


def infer(forest, T, la=None, psi=None, p=None,
          lower_bounds=DEFAULT_LOWER_BOUNDS, upper_bounds=DEFAULT_UPPER_BOUNDS, ci=False, **kwargs):
    """
    Infers BD model parameters from a given forest.

    :param forest: list of one or more trees
    :param la: transmission rate
    :param psi: removal rate
    :param p: sampling probability
    :param lower_bounds: array of lower bounds for parameter values (la, psi, p)
    :param upper_bounds: array of upper bounds for parameter values (la, psi, p)
    :param ci: whether to calculate the CIs or not
    :return: tuple(vs, cis) of estimated parameter values vs=[la, psi, p]
        and CIs ci=[[la_min, la_max], [psi_min, psi_max], [p_min, p_max]].
        In the case when CIs were not set to be calculated,
        their values would correspond exactly to the parameter values.
    """
    if la is None and psi is None and p is None:
        raise ValueError('At least one of the model parameters needs to be specified for identifiability')
    bounds = np.zeros((3, 2), dtype=np.float64)
    bounds[:, 0] = lower_bounds
    bounds[:, 1] = upper_bounds
    start_parameters = get_start_parameters(forest, la, psi, p)
    input_params = np.array([la, psi, p])
    print('Fixed input parameter(s): {}'
          .format(', '.join('{}={:g}'.format(*_)
                            for _ in zip(PARAMETER_NAMES[input_params != None], input_params[input_params != None]))))
    print('Starting BD parameters: {}'.format(start_parameters))
    vs, cis, lk = optimize_likelihood_params(forest, T, input_parameters=input_params,
                                             loglikelihood=loglikelihood, bounds=bounds[input_params == None],
                                             start_parameters=start_parameters, cis=ci)
    print('Estimated BD parameters: {}'.format(vs))
    if ci:
        print('Estimated CIs:\n{}'.format(cis))
    return vs, cis


def save_results(vs, cis, log, ci=False):
    os.makedirs(os.path.dirname(os.path.abspath(log)), exist_ok=True)
    with open(log, 'w+') as f:
        f.write(',{}\n'.format(','.join(['R0', 'infectious time', 'sampling probability',
                                         'transmission rate', 'removal rate'])))
        la, psi, rho = vs
        R0 = la / psi
        rt = 1 / psi
        (la_min, la_max), (psi_min, psi_max), (rho_min, rho_max) = cis
        R0_min, R0_max = la_min / psi, la_max / psi
        rt_min, rt_max = 1 / psi_max, 1 / psi_min
        f.write('value,{}\n'.format(','.join(str(_) for _ in [R0, rt, rho, la, psi])))
        if ci:
            f.write('CI_min,{}\n'.format(
                ','.join(str(_) for _ in [R0_min, rt_min, rho_min, la_min, psi_min])))
            f.write('CI_max,{}\n'.format(
                ','.join(str(_) for _ in [R0_max, rt_max, rho_max, la_max, psi_max])))


def main():
    """
    Entry point for tree parameter estimation with the BD model with command-line arguments.
    :return: void
    """
    import argparse

    parser = \
        argparse.ArgumentParser(description="Estimated BD parameters.")
    parser.add_argument('--nwk', required=True, type=str, help="input tree file")
    parser.add_argument('--la', required=False, default=None, type=float, help="transmission rate")
    parser.add_argument('--psi', required=False, default=None, type=float, help="removal rate")
    parser.add_argument('--p', required=False, default=None, type=float, help='sampling probability')
    parser.add_argument('--log', required=True, type=str, help="output log file")
    parser.add_argument('--upper_bounds', required=False, type=float, nargs=3,
                        help="upper bounds for parameters (la, psi, p)", default=DEFAULT_UPPER_BOUNDS)
    parser.add_argument('--lower_bounds', required=False, type=float, nargs=3,
                        help="lower bounds for parameters (la, psi, p)", default=DEFAULT_LOWER_BOUNDS)
    parser.add_argument('--ci', action="store_true", help="calculate the CIs")
    params = parser.parse_args()

    if params.la is None and params.psi is None and params.p is None:
        raise ValueError('At least one of the model parameters needs to be specified for identifiability')

    forest = read_forest(params.nwk)
    annotate_forest_with_time(forest)
    T = get_T(T=None, forest=forest)
    print('Read a forest of {} trees with {} tips in total, evolving over time {}'
          .format(len(forest), sum(len(_) for _ in forest), T))

    vs, cis = infer(forest, T, **vars(params))
    save_results(vs, cis, params.log, ci=params.ci)


def loglikelihood_main():
    """
    Entry point for tree likelihood estimation with the BD model with command-line arguments.
    :return: void
    """
    import argparse

    parser = \
        argparse.ArgumentParser(description="Calculate BD likelihood on a given forest for given parameter values.")
    parser.add_argument('--la', required=True, type=float, help="transmission rate")
    parser.add_argument('--psi', required=True, type=float, help="removal rate")
    parser.add_argument('--p', required=True, type=float, help='sampling probability')
    parser.add_argument('--nwk', required=True, type=str, help="input tree file")
    params = parser.parse_args()

    forest = read_forest(params.nwk)
    annotate_forest_with_time(forest)
    T = get_T(T=None, forest=forest)
    lk = loglikelihood(forest, la=params.la, psi=params.psi, rho=params.p, T=T)
    print(lk)


if '__main__' == __name__:
    main()
