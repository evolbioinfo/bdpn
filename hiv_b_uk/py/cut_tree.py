from pastml.tree import parse_nexus, annotate_dates, DATE


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


if '__main__' == __name__:
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--tree', type=str)
    parser.add_argument('--min_year', type=int, default=2012)
    parser.add_argument('--max_year', type=int, default=2015)
    parser.add_argument('--forest', type=str)
    params = parser.parse_args()

    tree = parse_nexus(params.tree)[0]
    annotate_dates([tree])

    tree, _ = cut_tree(tree, params.max_year + 1)
    _, forest = cut_tree(tree, params.min_year)

    with open(params.forest, 'w+') as f:
        for root in forest:
            f.write('{}\n'.format(root.write(format_root_node=True, format=5, features=[DATE])))
