import logging
import os

import numpy as np
from pastml.tree import read_forest, annotate_dates, DATE

from bdpn import bdpn, bd
from bdpn.model_distinguisher import pick_cherries, nonparametric_cherry_diff
from bdpn.parameter_estimator import optimize_likelihood_params, AIC
from bdpn.tree_manager import annotate_tree, TIME

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'trees'))

# gotree resolve -i D_CRF_19_timetree.nwk | gotree brlen setmin -l 0.0027397260273972603 -o D_CRF_19_timetree.resolved.nwk
NWK_CUBA = os.path.join(DATA_DIR, 'D_CRF_19_timetree.resolved.nwk')
min_cherry_num = 25


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


def check_cherries(forest, block_size=20, log=None):
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

        logging.info(
            'For {n} cherries with roots in [{start}-{stop}), PN test {res}.'
            .format(res=result, start=getattr(cherries[0].root, DATE), stop=getattr(cherries[-1].root, DATE),
                    n=len(cherries)))

    pval = sum(results) / len(results)

    logging.info("Total PN test {}.".format(pval))

    if log is not None:
        with open(log, 'w+') as f:
            f.write('Total\t{}\n'.format(pval))
            for years, val in zip(root_times, results):
                f.write('{}-{}\t{}\n'.format(*years, val))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")

    tree = read_forest(NWK_CUBA)[0]
    annotate_dates([tree], root_dates=[1974.0283399999998])
    annotate_tree(tree)
    forest = [tree]

    check_cherries(forest)
    cut_year = 2000
    root, forest = cut_tree(tree, cut_year)
    check_cherries(forest)

    bounds = np.array([[0.1, 12], [0.05, 1], [3, 365], [0.0001, 0.99999], [0.0001, 0.99999]])
    start_parameters = (bounds[:, 0] + bounds[:, 1]) / 2
    psi = 1 / 5
    input_parameters = np.array([None, psi, None, None, None])

    vs_bdpn, ci_bdpn = optimize_likelihood_params(forest, input_parameters=input_parameters,
                                                  loglikelihood=bdpn.loglikelihood,
                                                  bounds=bounds[input_parameters == None],
                                                  start_parameters=start_parameters, cis=True)
    lk_bdpn = bdpn.loglikelihood(forest, *vs_bdpn)
    print('Found BDPN params: {}\t loglikelihood: {}, AIC: {},\n\t CIs:\n\t{}'.format(vs_bdpn, lk_bdpn, AIC(4, lk_bdpn),
                                                                                      ci_bdpn))

    bd_indices = [0, 1, 3]
    input_parameters_bd = input_parameters[bd_indices]
    bounds_bd = bounds[bd_indices]
    start_parameters_bd = (bounds_bd[:, 0] + bounds_bd[:, 1]) / 2
    vs_bd, _ = optimize_likelihood_params(forest, input_parameters=input_parameters_bd,
                                          loglikelihood=bd.loglikelihood, bounds=bounds_bd[input_parameters_bd == None],
                                          start_parameters=start_parameters_bd)
    lk_bd = bd.loglikelihood(forest, *vs_bd)
    print('Found BD params: {}\t loglikelihood: {}, AIC: {}'.format(vs_bd, lk_bd, AIC(2, lk_bd)))
