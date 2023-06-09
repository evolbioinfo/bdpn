import pandas as pd
from ete3 import Tree

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Splits a tree into the root part and the bottom forest.")
    parser.add_argument('--in_nwk', required=True, type=str, help="Nwk tree to cut")
    parser.add_argument('--out_forest', type=str, help="Nwk containing a forest of cut subtrees")
    parser.add_argument('--out_root', type=str, help="Nwk containing a the root subtree")
    parser.add_argument('--threshold', type=float, help="Date at which to cut")
    params = parser.parse_args()

    tree = Tree(params.in_nwk)
    for n in tree.traverse('preorder'):
        parent_time = 0 if n.is_root() else getattr(n.up, 'time')
        n.add_feature('time', parent_time + n.dist)
    height = max(getattr(t, 'time') for t in tree)
    df.loc[0, 'T'] = height * (1 - params.threshold)
    height = height * params.threshold
    # tips	f	u	T	mu	la	psi	p
    f = 0
    tips = 0
    with open(params.out_nwk, 'w+') as file:
        todo = [tree]
        while todo:
            n = todo.pop()
            time = getattr(n, 'time')
            if time >= height:
                n.detach()
                n.dist = time - height
                file.write(n.write(format_root_node=True, format=5) + '\n')
                f += 1
                tips += len(n)
            else:
                todo.extend(n.children)
    df.loc[0, 'f'] = f
    df.loc[0, 'tips'] = tips
    rho = df.loc[0, 'p']
    df.loc[0, 'u'] = round(f / rho * (1 - rho))
    df.to_csv(params.out_log, sep='\t', index=False)
