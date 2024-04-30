import logging
import os

from treesimulator.generator import generate
from treesimulator.mtbd_models import BirthDeathModel, PNModel

from bdpn.model_distinguisher import pn_test, cherry_diff_plot, sign_test, estimate_pn
from bdpn.tree_manager import read_forest
from treesimulator import simulate_forest_bd

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'simulations', 'trees'))
NWK_BD = os.path.join(DATA_DIR, 'BD', 'tree.{}.nwk')
NWK_BDPN = os.path.join(DATA_DIR, 'BDPN', 'tree.{}.nwk')
min_cherry_num = 100

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")

    forest = read_forest('/home/azhukova/projects/bdpn/trees/D_CRF_19_timetree.resolved.nwk')
    print(sign_test(forest))
    # forest = read_forest('/home/azhukova/projects/bdpn/hiv_b_uk/data/forest.2012_2015.0.nwk')
    # print(sign_test(forest))
    # print(estimate_pn(forest))
    # exit()

    bd_model = BirthDeathModel(p=0.7, la=2, psi=1)
    forest, _, _ = generate(bd_model, min_tips=5000, max_tips=5050)
    cherry_diff_plot(forest, outfile='bd.jpg')
    print(sign_test(forest))
    print(estimate_pn(forest))
    forest[0].write(outfile='bd.nwk')

    pn_model = PNModel(model=bd_model, pn=0.9, removal_rate=25)
    forest, _, _ = generate(pn_model, min_tips=5000, max_tips=5050)
    cherry_diff_plot(forest, outfile='bdpn.jpg')
    print(sign_test(forest))
    print(estimate_pn(forest))
    forest[0].write(outfile='bdpn.nwk')

    # for tree_id in range(1):
    #     for nwk in (NWK_BD.format(tree_id), NWK_BDPN.format(tree_id)):
    #         forest = read_forest(nwk)
    #         cherry_diff_plot(forest)
    #         pval = pn_test(forest, repetitions=1e2)
    #         logging.info('Tree {tree}, PN test {res}.'.format(res=pval, tree=nwk.replace(DATA_DIR, '')))
    #         pval = sign_test(forest)
    #         logging.info('Tree {tree}, PN-sign test {res}.'.format(res=pval, tree=nwk.replace(DATA_DIR, '')))
