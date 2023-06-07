import logging
import os
from collections import defaultdict

from ete3 import Tree

from bdpn.model_distinguisher import pick_cherries, nonparametric_cherry_diff, pick_motifs
from bdpn.tree_manager import annotate_tree, TIME

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'trees'))
NWK_BD = os.path.join(DATA_DIR, 'bd', 'tree.1.nwk')
NWK_BDPN = os.path.join(DATA_DIR, 'bdpn', 'tree.1.nwk')
min_cherry_num = 50

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")

    for nwk in (NWK_BD, NWK_BDPN):
        logging.info('==============={}============'.format(nwk.replace(DATA_DIR, '')))
        tree = Tree(nwk)
        annotate_tree(tree)
        T = max(getattr(_, TIME) for _ in tree)
        dT = T / 100

        all_cherries = list(pick_cherries(tree, include_polytomies=True))
        logging.info('Picked {} cherries with {} roots.'.format(len(all_cherries), len({_.root for _ in all_cherries})))
        year2cherries = defaultdict(list)
        for _ in all_cherries:
            year2cherries[getattr(_.root, TIME) // dT].append(_)

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
