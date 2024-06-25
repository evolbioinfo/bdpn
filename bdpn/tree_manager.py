import os
import re
from datetime import datetime

from Bio import Phylo
from ete3 import Tree, TreeNode

TIME = 'time'

DATE_REGEX = r'[+-]*[\d]+[.\d]*(?:[e][+-][\d]+){0,1}'

NOTIFIERS = 'notifiers'


def read_nexus(tree_path):
    with open(tree_path, 'r') as f:
        nexus = f.read()
    # replace CI_date="2019(2018,2020)" with CI_date="2018 2020"
    nexus = re.sub(r'CI_date="({})\(({}),({})\)"'.format(DATE_REGEX, DATE_REGEX, DATE_REGEX), r'CI_date="\2 \3"',
                   nexus)
    temp = tree_path + '.{}.temp'.format(datetime.timestamp(datetime.now()))
    with open(temp, 'w') as f:
        f.write(nexus)
    trees = list(Phylo.parse(temp, 'nexus'))
    os.remove(temp)
    return trees


def parse_nexus(tree_path):
    trees = []
    for nex_tree in read_nexus(tree_path):
        todo = [(nex_tree.root, None)]
        tree = None
        while todo:
            clade, parent = todo.pop()
            dist = 0
            try:
                dist = float(clade.branch_length)
            except:
                pass
            name = getattr(clade, 'name', None)
            if not name:
                name = getattr(clade, 'confidence', None)
                if not isinstance(name, str):
                    name = None
            node = TreeNode(dist=dist, name=name)
            if parent is None:
                tree = node
            else:
                parent.add_child(node)
            todo.extend((c, node) for c in clade.clades)
        trees.append(tree)
    return trees


def read_tree(tree_path):
    tree = None
    for f in (3, 2, 5, 0, 1, 4, 6, 7, 8, 9):
        try:
            tree = Tree(tree_path, format=f)
            break
        except:
            continue
    if not tree:
        raise ValueError('Could not read the tree {}. Is it a valid newick?'.format(tree_path))
    return tree


def read_forest(tree_path):
    try:
        roots = parse_nexus(tree_path)
        if roots:
            return roots
    except:
        pass
    with open(tree_path, 'r') as f:
        nwks = f.read().replace('\n', '').split(';')
    if not nwks:
        raise ValueError('Could not find any trees (in newick or nexus format) in the file {}.'.format(tree_path))
    return [read_tree(nwk + ';') for nwk in nwks[:-1]]


def annotate_tree(tree):
    for n in tree.traverse('preorder'):
        p_time = 0 if n.is_root() else getattr(n.up, TIME)
        n.add_feature(TIME, p_time + n.dist)


def annotate_forest_with_time(forest):
    for tree in forest:
        if not hasattr(tree, TIME):
            annotate_tree(tree)


def get_T(T, forest):
    if T is None:
        T = 0
        for tree in forest:
            T = max(T, max(getattr(_, TIME) for _ in tree))
    return T


def sort_tree(tree):
    """
    Reorganise the tree in such a way that the oldest tip (with the minimal time) is always on the left.
    The tree must be time-annotated.

    :param tree:
    :return:
    """
    for n in tree.traverse('postorder'):
        ot_feature = 'oldest_tip'
        if n.is_leaf():
            n.add_feature(ot_feature, getattr(n, TIME))
            continue
        c1, c2 = n.children
        t1, t2 = getattr(c1, ot_feature), getattr(c2, ot_feature)
        if t1 > t2:
            n.children = [c2, c1]
        delattr(c1, ot_feature)
        delattr(c2, ot_feature)
        if not n.is_root():
            n.add_feature(ot_feature, min(t1, t2))


def get_total_num_notifiers(tree):
    return sum(len(getattr(_, NOTIFIERS)) for _ in tree.traverse())


def get_max_num_notifiers(tree):
    n = len(tree)
    return n ** 2 - 2 * n + 2


def get_min_num_notifiers(tree):
    n = len(tree)
    return n



def preannotate_notifiers(forest):
    """
    Preannotates each tree node with potential notifiers from upper subtree
    :param forest: forest of trees to be annotated
    :return: void, adds NOTIFIERS feature to forest tree nodes.
        This feature contains a (potentially empty) set of upper tree notifiers
    """
    for tree in forest:
        for tip in tree:
            if not tip.is_root():
                parent = tip.up
                for sis in parent.children:
                    if sis != tip:
                        sis.add_feature(NOTIFIERS, getattr(sis, NOTIFIERS, set()) | {tip})
        tree.add_feature(NOTIFIERS, set())
        for node in tree.traverse('preorder'):
            notifiers = getattr(node, NOTIFIERS)
            for child in node.children:
                child.add_feature(NOTIFIERS, getattr(child, NOTIFIERS, set()) | notifiers)
