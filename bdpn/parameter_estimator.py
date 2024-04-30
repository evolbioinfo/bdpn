import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2

from bdpn.tree_manager import TIME, annotate_forest_with_time

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
        return 0
    min_lh_value = np.min(non_zero_loglh_array)
    max_lh_value = np.max(non_zero_loglh_array)

    factors = 0
    if max_lh_value > max_limit - 2:
        factors = max_limit - max_lh_value - 2
    elif min_lh_value < min_limit + 2:
        factors = min(min_limit - min_lh_value + 2, max_limit - max_lh_value - 2)
    loglikelihood_array += factors
    return factors


def optimize_likelihood_params(forest, input_parameters, loglikelihood, bounds, start_parameters, cis=False, threads=1):
    """
    Optimizes the likelihood parameters for a given forest and a given MTBD model.


    :param forest: a list of ete3.Tree trees
    :return: tuple: (the values of optimized parameters, CIs)
    """
    # print('Bounds are set to {}'.format(bounds))
    annotate_forest_with_time(forest)
    T = 0
    for tree in forest:
        T = max(T, max(getattr(_, TIME) for _ in tree))

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
        res = loglikelihood(forest, *ps, T, threads=threads)
        # print("{}\t-->\t{:g}".format(ps, res))
        return -res

    x0 = get_optimised_params_from_real(start_parameters)
    best_log_lh = -get_v(x0)

    for i in range(5):
        if i == 0:
            vs = x0
        else:
            vs = np.random.uniform(bounds[:, 0], bounds[:, 1])

        fres = minimize(get_v, x0=vs, method='L-BFGS-B', bounds=bounds)
        if fres.success and not np.any(np.isnan(fres.x)):
            if -fres.fun >= best_log_lh:
                x0 = np.array(fres.x)
                best_log_lh = -fres.fun
                break
            # print('Attempt {} of trying to optimise the parameters: {} -> {}.'.format(i, x0, -fres.fun))
    optimised_parameters = get_real_params_from_optimised(x0)

    n = len(optimised_parameters)
    optimised_cis = np.zeros((n, 2))
    optimised_cis[:, 0] = np.array(optimised_parameters)
    optimised_cis[:, 1] = np.array(optimised_parameters)

    if cis:
        lk_threshold = loglikelihood(forest, *optimised_parameters, T, threads=threads) - chi2.ppf(q=0.95, df=1) / 2

        def binary_search(v_min, v_max, i, lower=True):
            rps = np.array(optimised_parameters)
            v = v_min + (v_max - v_min) / 2
            if (v_max - v_min) < 1e-4:
                return v
            rps[i] = v
            lk = loglikelihood(forest, *rps, T, threads=threads) #threads=max(1, threads // len(bounds)))
            lk_diff = lk - lk_threshold
            if np.abs(lk_diff) < 1e-4:
                return v

            go_left = (lower and lk_diff > 0) or ((not lower) and lk_diff < 0)

            if go_left:
                return binary_search(v_min, v, i, lower)
            return binary_search(v, v_max, i, lower)

        def get_ci(args):
            (i, optimised_value), (b_min, b_max) = args
            optimised_cis[i, 0] = binary_search(b_min, optimised_value, i, True)
            optimised_cis[i, 1] = binary_search(optimised_value, b_max, i, False)

        # if threads > 1:
        #     with ThreadPool(processes=min(threads, len(bounds))) as pool:
        #         pool.map(func=get_ci,
        #                  iterable=zip(((i, v) for (i, v) in enumerate(optimised_parameters)
        #                                if input_parameters[i] is not None), bounds), chunksize=1)
        # else:
        for _ in zip(((i, v) for (i, v) in enumerate(optimised_parameters) if input_parameters[i] is None),
                     bounds):
            get_ci(_)

    return optimised_parameters, optimised_cis


def AIC(k, loglk):
    return 2 * k - 2 * loglk