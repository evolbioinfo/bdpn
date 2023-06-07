import numpy as np
from scipy.optimize import minimize

from bdpn.tree_manager import TIME, annotate_tree

MIN_VALUE = np.log(np.finfo(np.float64).eps)
MAX_VALUE = np.log(np.finfo(np.float64).max)


def rescale_log(loglikelihood_array):
    """
    Rescales the likelihood array if it gets too small/large, by multiplying it by a factor of e.
    :param loglikelihood_array: numpy array containing the loglikelihood to be rescaled
    :return: float, factor of e by which the likelihood array has been multiplies.
    """

    max_limit = MAX_VALUE
    min_limit = MIN_VALUE

    non_zero_loglh_array = loglikelihood_array[loglikelihood_array > -np.inf]
    if len(non_zero_loglh_array) == 0:
        raise ValueError('Underflow')
    min_lh_value = np.min(non_zero_loglh_array)
    max_lh_value = np.max(non_zero_loglh_array)

    factors = 0
    if max_lh_value > max_limit - 2:
        factors = max_limit - max_lh_value - 2
    elif min_lh_value < min_limit + 2:
        factors = min(min_limit - min_lh_value + 2, max_limit - max_lh_value - 2)
    loglikelihood_array += factors
    return factors


def optimize_likelihood_params(tree, input_parameters, loglikelihood, bounds, start_parameters):
    """
    Optimizes the likelihood parameters for a given forest and a given MTBD model.


    :param forest: a list of ete3.Tree trees, annotated with node states and times via features STATE_K and TI.
    :param T: time at end of the sampling period
    :param model: MTBD model containing starting parameter values
    :param optimise: MTBD model whose rates indicate which parameters need to optimized:
        positive rates correspond to optimized parameters
    :param u: number of hidden trees, where no tip got sampled
    :return: the values of optimized parameters and the corresponding loglikelihood: (MU, LA, PSI, RHO, best_log_lh)
    """
    if not hasattr(tree, TIME):
        annotate_tree(tree)
    T = max(getattr(_, TIME) for _ in tree)
    # bounds, start_parameters = get_bounds_start(tree, *input_parameters)
    # print('Bounds are {}'.format(bounds))
    # print('Starting parameter values are {}'.format(start_parameters))

    def get_real_params_from_optimised(ps):
        ps = np.maximum(np.minimum(ps, bounds[:, 1]), bounds[:, 0])
        result = np.zeros(len(input_parameters))

        i = 0
        for par_index, par in enumerate(input_parameters):
            if par is None:
                result[par_index] = ps[i]
                i += 1
            else:
                result[par_index] = par
        return result

    def get_optimised_params_from_real(ps):
        result = []
        for par_index, par in enumerate(input_parameters):
            if par is None:
                result.append(ps[par_index])
        return np.array(result)

    def get_v(ps):
        if np.any(np.isnan(ps)):
            return np.nan
        ps = get_real_params_from_optimised(ps)
        res = loglikelihood(tree, *ps, T)
        # print("{}\t-->\t{:g}".format(ps, res))
        return -res

    x0 = get_optimised_params_from_real(start_parameters)
    best_log_lh = -get_v(x0)

    for i in range(10):
        if i == 0:
            vs = x0
        else:
            keep_searching = True
            while keep_searching:
                keep_searching = False
                vs = np.random.uniform(bounds[:, 0], bounds[:, 1])

        fres = minimize(get_v, x0=vs, method='L-BFGS-B', bounds=bounds)
        if fres.success and not np.any(np.isnan(fres.x)):
            if -fres.fun >= best_log_lh:
                x0 = np.array(fres.x)
                best_log_lh = -fres.fun
                break
        # print('Attempt {} of trying to optimise the parameters: {}.'.format(i, -fres.fun))
    return get_real_params_from_optimised(x0)


def get_rough_rate_etimates(tree):
    l = []
    for n in tree.traverse('preorder'):
        if n.dist:
            l.append(n.dist)
    max_rate = 10 / np.mean(l)
    min_rate = 1 / np.max(l)
    # print('Considering ', max_rate, ' as max rate and ', min_rate, ' as min rate')
    avg_rate = (min_rate + max_rate) / 2
    return avg_rate, max_rate, min_rate


def AIC(k, loglk):
    return 2 * k - 2 * loglk