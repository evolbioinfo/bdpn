import numpy as np
from scipy.optimize import minimize
from scipy.stats import stats

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


def optimize_likelihood_params(forest, input_parameters, loglikelihood, bounds, start_parameters, cis=False):
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
    for tree in forest:
        if not hasattr(tree, TIME):
            annotate_tree(tree)
    T = 0
    for tree in forest:
        T = max(T, max(getattr(_, TIME) for _ in tree))
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
        res = loglikelihood(forest, *ps, T)
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
    optimised_parameters = get_real_params_from_optimised(x0)

    n = len(optimised_parameters)
    optimised_cis = np.zeros((n, 2))
    optimised_cis[:, 0] = np.array(optimised_parameters)
    optimised_cis[:, 1] = np.array(optimised_parameters)

    if cis:
        lk_threshold = loglikelihood(forest, *optimised_parameters, T) - stats.chi2.ppf(q=0.95, df=1)

        def binary_search(v_min, v_max, i, lower=True):
            rps = np.array(optimised_parameters)
            v = v_min + (v_max - v_min) / 2
            if (v_max - v_min) < 1e-6:
                return v
            rps[i] = v
            lk_diff = loglikelihood(forest, *rps, T) - lk_threshold
            if np.abs(lk_diff) < 1e-6:
                return v

            go_left = (lower and lk_diff > 0) or ((not lower) and lk_diff < 0)

            if go_left:
                return binary_search(v_min, v, i)
            return binary_search(v, v_max, i)

        skipped_i = 0
        for i in range(n):
            optimised_value = optimised_parameters[i]
            if input_parameters[i] is not None:
                skipped_i += 1
                continue

            b_min, b_max = bounds[i - skipped_i, :]

            optimised_cis[i, 0] = binary_search(b_min, optimised_value, i, True)
            optimised_cis[i, 1] = binary_search(optimised_value, b_max, i, False)

    return optimised_parameters, optimised_cis


def AIC(k, loglk):
    return 2 * k - 2 * loglk