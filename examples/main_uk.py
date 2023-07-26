import logging
import os
from collections import defaultdict

import numpy as np
from matplotlib.pyplot import plot, show
from pastml.tree import read_forest, annotate_dates, DATE, parse_nexus

from bdpn import bdpn, bd
from bdpn.model_distinguisher import pick_cherries, nonparametric_cherry_diff
from bdpn.parameter_estimator import optimize_likelihood_params, AIC
from bdpn.tree_manager import annotate_tree, TIME

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'trees'))

# min len of 1 h
# lsd2 -i rooted_raxml.nwk -d lsd2.interval.dates -e 3 -s 3102 -u 0.0027397260273972603 -l 0 -m 20000
NWK_UK = os.path.join(DATA_DIR, 'UK', 'timetree.nexus')
PN_TEST_LOG = os.path.join(DATA_DIR, 'UK', 'cherry_test.txt')
min_cherry_num = 50


def save_estimates(log, vs, cis):
    with open(log, 'w+') as f:
        f.write('\t{}\n'.format(','.join(['R0', 'infectious time', 'sampling probability', 'notification probability',
                                          'removal time after notification',
                                          'transmission rate', 'removal rate', 'partner removal rate'])))
        la, psi, psi_p, rho, rho_p = vs
        R0 = la / psi
        rt = 1 / psi
        prt = 1 / psi_p
        (la_min, la_max), (psi_min, psi_max), (psi_p_min, psi_p_max), (rho_min, rho_max), (rho_p_min, rho_p_max) = cis
        R0_min, R0_max = la_min / psi, la_max / psi
        rt_min, rt_max = 1 / psi_max, 1 / psi_min
        prt_min, prt_max = 1 / psi_p_max, 1 / psi_p_min
        f.write('value,{}\n'.format(','.join(str(_) for _ in [R0, rt, rho, rho_p, prt, la, psi, psi_p])))
        f.write('CI_min,{}\n'.format(
            ','.join(str(_) for _ in [R0_min, rt_min, rho_min, rho_p_min, prt_min, la_min, psi_min, psi_p_min])))
        f.write('CI_max,{}\n'.format(
            ','.join(str(_) for _ in [R0_max, rt_max, rho_max, rho_p_max, prt_max, la_max, psi_max, psi_p_max])))


def cut_tree(tree, threshold_date):
    forest = []
    todo = [tree]
    while todo:
        n = todo.pop()
        date = getattr(n, DATE)
        if date >= threshold_date:
            parent = n.up
            n.detach()
            n.dist = date - threshold_date
            forest.append(n)
            if parent and len(parent.children) == 1:
                child = parent.children[0]
                if not parent.is_root():
                    grandparent = parent.up
                    grandparent.remove_child(parent)
                    grandparent.add_child(child, dist=child.dist + parent.dist)
                else:
                    child.dist += parent.dist
                    child.detach()
                    tree = child
        else:
            todo.extend(n.children)
    print("Cut the tree into a root tree of {} tips and {} {}-on trees of {} tips in total"
          .format(len(tree), len(forest), threshold_date, sum(len(_) for _ in forest)))
    return tree, forest


def check_cherries(forest, block_size=100, log=None):
    all_cherries = []
    for tree in forest:
        all_cherries.extend(pick_cherries(tree, include_polytomies=True))
    all_cherries = sorted(all_cherries, key=lambda _: getattr(_.root, TIME))

    logging.info('Picked {} cherries with {} roots.'.format(len(all_cherries), len({_.root for _ in all_cherries})))

    results = []
    root_times = []
    for i in range(0, len(all_cherries) // block_size):
        cherries = all_cherries[i * block_size: (i + 1) * block_size]
        result = nonparametric_cherry_diff(cherries, repetitions=1e3)
        results.append(result)
        root_times.append((getattr(cherries[0].root, DATE), getattr(cherries[-1].root, DATE)))

        # logging.info(
        #     'For {n} cherries with roots in [{start}-{stop}), PN test {res}.'
        #     .format(res=result, start=getattr(cherries[0].root, DATE), stop=getattr(cherries[-1].root, DATE),
        #             n=len(cherries)))

    pval = sum(results) / len(results)

    logging.info("Total PN test {}.".format(pval))

    if log is not None:
        with open(log, 'w+') as f:
            f.write('Total\t{}\n'.format(pval))
            for years, val in zip(root_times, results):
                f.write('{}-{}\t{}\n'.format(*years, val))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")

    tree = parse_nexus(NWK_UK)[0]
    annotate_dates([tree])
    annotate_tree(tree)

    # check_cherries([tree], log=PN_TEST_LOG)
    cut_year = 2012
    root, forest = cut_tree(tree, cut_year)
    # check_cherries(forest)

    # psi_n = np.arange(0.5, 10000000, 10000)
    # lks = [bdpn.loglikelihood(forest, la=0.295, psi=0.194, psi_n=_, rho=0.400, rho_n=0.033) for _ in psi_n]
    #
    # plot(psi_n, lks)
    # show()

    # for psi in (0.1, 0.15, 0.18, 0.2):
    #     print("=======psi={}============".format(psi))
    #     bounds = np.array([[0.1, 12], [0.05, 1], [0.5, 365], [0.0001, 0.99999], [0.0001, 0.99999]])
    #     start_parameters = (bounds[:, 0] + bounds[:, 1]) / 2
    #     input_parameters = np.array([None, psi, None, None, None])
    #     vs_bdpn, ci_bdpn = optimize_likelihood_params(forest, input_parameters=input_parameters,
    #                                                   loglikelihood=bdpn.loglikelihood,
    #                                                   bounds=bounds[input_parameters == None],
    #                                                   start_parameters=start_parameters, cis=True)
    #     lk_bdpn = bdpn.loglikelihood(forest, *vs_bdpn)
    #     print('Found BDPN params for {}-on trees: {}\t loglikelihood: {}, AIC: {}\n\t CIs:\n\t{}'
    #           .format(cut_year, ', '.join('{:.3f}'.format(_) for _ in vs_bdpn), lk_bdpn, AIC(4, lk_bdpn), ci_bdpn))
    #     save_estimates(NWK_UK.replace('.nwk', '2005.{}={}.est'.format(cut_year, psi)), vs_bdpn, ci_bdpn)


    for rho in (0.5, 0.6, 0.7):
        print("=======rho={}============".format(rho))
        bounds = np.array([[0.1, 12], [0.05, 1], [0.5, 365], [0.0001, 0.99999], [0.0001, 0.99999]])
        start_parameters = (bounds[:, 0] + bounds[:, 1]) / 2
        input_parameters = np.array([None, None, None, rho, None])
        vs_bdpn, ci_bdpn = optimize_likelihood_params(forest, input_parameters=input_parameters,
                                                      loglikelihood=bdpn.loglikelihood,
                                                      bounds=bounds[input_parameters == None],
                                                      start_parameters=start_parameters, cis=True)
        lk_bdpn = bdpn.loglikelihood(forest, *vs_bdpn)
        print('Found BDPN params for {}-on trees: {}\t loglikelihood: {}, AIC: {}\n\t CIs:\n\t{}'
              .format(cut_year, ', '.join('{:.3f}'.format(_) for _ in vs_bdpn), lk_bdpn, AIC(4, lk_bdpn), ci_bdpn))
        save_estimates(NWK_UK.replace('.nwk', '{}.rho={}.est'.format(cut_year, rho)), vs_bdpn, ci_bdpn)
        # input_parameters_root = np.array([None, vs_bdpn[1], None, None, None])
        # vs_bdpn, ci_bdpn = optimize_likelihood_params([root], input_parameters=input_parameters_root,
        #                                      loglikelihood=bdpn.loglikelihood,
        #                                      bounds=bounds[input_parameters_root == None],
        #                                      start_parameters=start_parameters, cis=True)
        # lk_bdpn = bdpn.loglikelihood([root], *vs_bdpn)
        # print('Found BDPN params for the root tree: {}\t loglikelihood: {}, AIC: {}\n\t CIs:\n\t{}'
        #       .format(', '.join('{:.3f}'.format(_) for _ in vs_bdpn), lk_bdpn, AIC(4, lk_bdpn), ci_bdpn))
        #
        # bd_indices = [0, 1, 3]
        # input_parameters_bd = input_parameters[bd_indices]
        # bounds_bd = bounds[bd_indices]
        # start_parameters_bd = (bounds_bd[:, 0] + bounds_bd[:, 1]) / 2
        # vs_bd, ci_bd = optimize_likelihood_params(forest, input_parameters=input_parameters_bd,
        #                                    loglikelihood=bd.loglikelihood,
        #                                    bounds=bounds_bd[input_parameters_bd == None],
        #                                    start_parameters=start_parameters_bd, cis=True)
        # lk_bd = bd.loglikelihood(forest, *vs_bd)
        # print('Found BD params for {}-on trees: {}\t loglikelihood: {}, AIC: {}\n\t CIs:\n\t{}'
        #       .format(cut_year, ', '.join('{:.3f}'.format(_) for _ in vs_bd), lk_bd, AIC(2, lk_bd), ci_bd))
        # input_parameters_bd_root = np.array([None, vs_bdpn[1], None])
        # vs_bd, ci_bd = optimize_likelihood_params([root], input_parameters=input_parameters_bd_root,
        #                                    loglikelihood=bd.loglikelihood,
        #                                    bounds=bounds_bd[input_parameters_bd_root == None],
        #                                    start_parameters=start_parameters_bd, cis=True)
        # lk_bd = bd.loglikelihood([root], *vs_bd)
        # print('Found BD params for the root tree: {}\t loglikelihood: {}, AIC: {}\n\t CIs:\n\t{}'
        #       .format(', '.join('{:.3f}'.format(_) for _ in vs_bd), lk_bd, AIC(2, lk_bd), ci_bd))

