import logging
import os
from collections import defaultdict

import numpy as np
from pastml.tree import read_forest, annotate_dates, DATE

from bdpn import bdpn, bd
from bdpn.model_distinguisher import pick_cherries, nonparametric_cherry_diff, pick_motifs
from bdpn.parameter_estimator import optimize_likelihood_params, AIC
from bdpn.tree_manager import annotate_tree

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'trees'))

# gotree resolve -i UK_B_raxml.lsd2.nwk | gotree brlen setmin -l 0.0027397260273972603 -o UK_B_raxml.lsd2.resolved.nwk
NWK_UK = os.path.join(DATA_DIR, 'UK_B_raxml.lsd2.resolved.nwk')
min_cherry_num = 50


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
            if len(parent.children) == 1:
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


def check_cherries(tree):
    all_cherries = list(pick_cherries(tree, include_polytomies=True))
    logging.info('Picked {} cherries with {} roots.'.format(len(all_cherries), len({_.root for _ in all_cherries})))
    year2cherries = defaultdict(list)
    for _ in all_cherries:
        year2cherries[int(getattr(_.root, DATE))].append(_)
    total_cherries = len(all_cherries)
    year = min(year2cherries.keys())
    cherries = []
    max_year = max(year2cherries.keys())
    while year <= max_year:
        start_year = year
        while len(cherries) < min_cherry_num and year <= max_year:
            cherries.extend(year2cherries[year])
            year += 1
        # drop the last few cherries if they are not the only ones we have
        if len(cherries) < min_cherry_num / 2 and min_cherry_num <= total_cherries:
            break
        result = nonparametric_cherry_diff(cherries, repetitions=1e3)

        logging.info(
            'For {n} cherries with a root in {start}-{stop}, PN test {res}.'
            .format(res=result, start=start_year, stop=year - 1, n=len(cherries)))
        cherries = []

        # res = community_tracing_test(tree)
        # logging.info('CT test: {}'.format(res))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")

    tree = read_forest(NWK_UK)[0]
    annotate_dates([tree])
    annotate_tree(tree)

    # check_cherries(tree)
    cut_year = 2005
    root, forest = cut_tree(tree, cut_year)

    for psi in (1/3, 1/5, 1/7, 1/9):
        print("=======psi={}============".format(psi))
        bounds = np.array([[0.1, 12], [0.05, 1], [0.5, 365], [0.0001, 0.99999], [0.0001, 0.99999]])
        start_parameters = (bounds[:, 0] + bounds[:, 1]) / 2
        input_parameters = np.array([None, psi, None, None, None])
        vs_bdpn, ci_bdpn = optimize_likelihood_params(forest, input_parameters=input_parameters,
                                             loglikelihood=bdpn.loglikelihood,
                                             bounds=bounds[input_parameters == None],
                                             start_parameters=start_parameters, cis=True)
        lk_bdpn = bdpn.loglikelihood(forest, *vs_bdpn)
        print('Found BDPN params for {}-on trees: {}\t loglikelihood: {}, AIC: {}\n\t CIs:\n\t{}'
              .format(cut_year, ', '.join('{:.3f}'.format(_) for _ in vs_bdpn), lk_bdpn, AIC(4, lk_bdpn), ci_bdpn))
        vs_bdpn, ci_bdpn = optimize_likelihood_params([root], input_parameters=input_parameters,
                                             loglikelihood=bdpn.loglikelihood,
                                             bounds=bounds[input_parameters == None],
                                             start_parameters=start_parameters, cis=True)
        lk_bdpn = bdpn.loglikelihood([root], *vs_bdpn)
        print('Found BDPN params for the root tree: {}\t loglikelihood: {}, AIC: {}\n\t CIs:\n\t{}'
              .format(', '.join('{:.3f}'.format(_) for _ in vs_bdpn), lk_bdpn, AIC(4, lk_bdpn), ci_bdpn))

        bd_indices = [0, 1, 3]
        input_parameters_bd = input_parameters[bd_indices]
        bounds_bd = bounds[bd_indices]
        start_parameters_bd = (bounds_bd[:, 0] + bounds_bd[:, 1]) / 2
        vs_bd, ci_bd = optimize_likelihood_params(forest, input_parameters=input_parameters_bd,
                                           loglikelihood=bd.loglikelihood,
                                           bounds=bounds_bd[input_parameters_bd == None],
                                           start_parameters=start_parameters_bd, cis=True)
        lk_bd = bd.loglikelihood(forest, *vs_bd)
        print('Found BD params for {}-on trees: {}\t loglikelihood: {}, AIC: {}\n\t CIs:\n\t{}'
              .format(cut_year, ', '.join('{:.3f}'.format(_) for _ in vs_bd), lk_bd, AIC(2, lk_bd), ci_bd))
        vs_bd, ci_bd = optimize_likelihood_params([root], input_parameters=input_parameters_bd,
                                           loglikelihood=bd.loglikelihood,
                                           bounds=bounds_bd[input_parameters_bd == None],
                                           start_parameters=start_parameters_bd, cis=True)
        lk_bd = bd.loglikelihood([root], *vs_bd)
        print('Found BD params for the root tree: {}\t loglikelihood: {}, AIC: {}\n\t CIs:\n\t{}'
              .format(', '.join('{:.3f}'.format(_) for _ in vs_bd), lk_bd, AIC(2, lk_bd), ci_bd))

    exit()

    with open(os.path.join(DATA_DIR, "motifs.nwk"), "w+") as f:
        for motif in pick_motifs(tree, pn_lookback=100, pn_delay=.5):
            if len(motif.clustered_tips) < 2:
                continue
            print(motif.root.get_ascii(attributes=["name", "dist"]))
            for tip, indices in motif.notified2index_list.items():
                print("tip {} was notified by {}".format(tip.name, [_.name for _ in indices]))
            print("")

            f.write("{}\n".format(motif.root.write(format=3)))
