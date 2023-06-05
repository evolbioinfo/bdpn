TIME = 'time'


def annotate_tree(tree):
    for n in tree.traverse('preorder'):
        if n.is_root():
            p_time = 0
        else:
            p_time = getattr(n.up, TIME)
        n.add_feature(TIME, p_time + n.dist)


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
