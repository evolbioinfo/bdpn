import logging
import os

from ete3 import Tree

from bdpn.model_distinguisher import pick_cherries, nonparametric_cherry_diff, pick_motifs
from bdpn.tree_manager import annotate_tree, TIME

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'trees'))
NWK_BD = os.path.join(DATA_DIR, 'bd', 'tree.1.nwk')
NWK_BDPN = os.path.join(DATA_DIR, 'bdpn', 'tree.1.nwk')
min_cherry_num = 100

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")

    for nwk in (NWK_BD, NWK_BDPN):
        logging.info('==============={}============'.format(nwk.replace(DATA_DIR, '')))

        results = []
        tree = Tree(nwk)
        annotate_tree(tree)

        all_cherries = sorted(list(pick_cherries(tree, include_polytomies=True)), key=lambda _: getattr(_.root, TIME))
        logging.info('Picked {} cherries with {} roots.'.format(len(all_cherries), len({_.root for _ in all_cherries})))
        for i in range(0, len(all_cherries) // min_cherry_num):
            cherries = all_cherries[i * min_cherry_num: (i + 1) * min_cherry_num]
            result = nonparametric_cherry_diff(cherries, repetitions=1e3)
            results.append(result)

            logging.info(
                'For {n} cherries with roots in [{start}-{stop}), PN test {res}.'
                .format(res=result, start=getattr(cherries[0].root, TIME), stop=getattr(cherries[-1].root, TIME), n=len(cherries)))
            cherries = []

        print(sum(results) / len(results))
