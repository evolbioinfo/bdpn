import glob

import numpy as np
from ete3 import Tree

from bdpn.bdpn_model import preannotate_notifiers
from bdpn.tree_manager import get_total_num_notifiers

nwks = glob.glob("/home/azhukova/Evolbioinfo/users/azhukova/projects/bdpn/simulations/medium/BDPN/tree.*.nwk")
nwks_bd = glob.glob("/home/azhukova/Evolbioinfo/users/azhukova/projects/bdpn/simulations/medium/BD/tree.*.nwk")


def get_ns(nwks):
    ns = []

    for nwk in nwks:
        tree = Tree(nwk)
        preannotate_notifiers([tree])
        n = get_total_num_notifiers(tree) / len(tree)
        ns.append(n)
    ns = np.array(ns)
    print(ns.min(), ns.max(), ns.mean(), np.quantile(ns, 0.25), np.quantile(ns, 0.5), np.quantile(ns, 0.75))
    return ns

ns = get_ns(nwks)
ns1 = get_ns(nwks_bd)


