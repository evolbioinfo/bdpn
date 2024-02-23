import logging
import os

from bdpn.model_distinguisher import pn_test
from bdpn.tree_manager import read_forest

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'simulations', 'trees'))
NWK_BD = os.path.join(DATA_DIR, 'BD', 'tree.{}.nwk')
NWK_BDPN = os.path.join(DATA_DIR, 'BDPN', 'tree.{}.nwk')
min_cherry_num = 100

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")

    for tree_id in range(100):
        for nwk in (NWK_BD.format(tree_id), NWK_BDPN.format(tree_id)):
            pval = pn_test(read_forest(nwk), repetitions=1e2)
            logging.info('Tree {tree}, PN test {res}.'.format(res=pval, tree=nwk.replace(DATA_DIR, '')))
