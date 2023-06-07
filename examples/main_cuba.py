import logging
import os
from collections import defaultdict

from pastml.tree import read_forest, annotate_dates, DATE

from bdpn import bdpn, bd
from bdpn.model_distinguisher import pick_cherries, nonparametric_cherry_diff, pick_motifs
from bdpn.parameter_estimator import optimize_likelihood_params, AIC
from bdpn.tree_manager import annotate_tree

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'trees'))

# gotree resolve -i D_CRF_19_timetree.nwk | gotree brlen setmin -l 0.0027397260273972603 -o D_CRF_19_timetree.resolved.nwk
NWK_CUBA = os.path.join(DATA_DIR, 'D_CRF_19_timetree.resolved.nwk')
min_cherry_num = 50

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")

    tree = read_forest(NWK_CUBA)[0]
    annotate_dates([tree], root_dates=[1974.0283399999998])
    annotate_tree(tree)

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
        result = nonparametric_cherry_diff(cherries, repetitions=1e4)

        logging.info(
            'For {n} cherries with a root in {start}-{stop}, PN test {res}.'
            .format(res=result, start=start_year, stop=year - 1, n=len(cherries)))
        cherries = []

        # res = community_tracing_test(tree)
        # logging.info('CT test: {}'.format(res))

    vs_bdpn = optimize_likelihood_params(tree, input_parameters=[None, 1/3, None, None, None],
                                         loglikelihood=bdpn.loglikelihood, get_bounds_start=bdpn.get_bounds_start)
    lk_bdpn = bdpn.loglikelihood(tree, *vs_bdpn)
    print('Found BDPN params: {}\t loglikelihood: {}, AIC: {}'.format(vs_bdpn, lk_bdpn, AIC(4, lk_bdpn)))

    vs_bd = optimize_likelihood_params(tree, input_parameters=[None, 1/3, None],
                                       loglikelihood=bd.loglikelihood, get_bounds_start=bd.get_bounds_start)
    lk_bd = bd.loglikelihood(tree, *vs_bd)
    print('Found BD params: {}\t loglikelihood: {}, AIC: {}'.format(vs_bd, lk_bd, AIC(2, lk_bd)))

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
