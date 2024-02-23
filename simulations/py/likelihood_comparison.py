import pandas as pd
from ete3 import Tree

from bdpn.bdpn import loglikelihood
import numpy as np
from scipy.stats import binomtest

PYBDEI = 'PyBDEI'

REAL_TYPE = 'real'


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Calculates likelihoods.")
    parser.add_argument('--estimates', type=str, help="estimated parameter", required=True)
    parser.add_argument('--tree_pattern', type=str, help="trees", required=True)
    parser.add_argument('--likelihoods', type=str, help="likelihood table")
    parser.add_argument('--log', type=str, help="likelihood stats")
    params = parser.parse_args()

    df = pd.read_csv(params.estimates, sep='\t', index_col=0)
    NON_REAL_TYPES = [_ for _ in df['type'].unique() if _ != REAL_TYPE]
    ALL_TYPES = NON_REAL_TYPES + [REAL_TYPE]

    df.index = df['type'] + '_' + df.index.map(str)
    lk_df, index = [], []
    for i in range(len(df) // len(ALL_TYPES)):
        tree = Tree(params.tree_pattern.format(i))
        p = df.loc['real_{}'.format(i), 'p']
        type2lk = {}
        for type in ALL_TYPES:
            la, psi, psi_n, rho, rho_n = df.loc['{}_{}'.format(type, i), ['lambda', 'psi', 'psi_p', 'p', 'pn']]
            type2lk[type] = loglikelihood([tree], la, psi, psi_n, rho, rho_n)
            index.append(i)
        best_lk = max(type2lk.values())
        for _ in ALL_TYPES:
            lk_df.append([_, type2lk[_], best_lk - type2lk[_], type2lk[_] - type2lk['real']])
    lk_df = pd.DataFrame(data=lk_df, columns=['type', 'loglk', 'diff with max', 'diff with real'], index=index)
    lk_df.to_csv(params.likelihoods, sep='\t')

    with open(params.log, 'w+') as f:

        for i in range(len(ALL_TYPES)):
            type1 = ALL_TYPES[i]
            log_lk_t1 = lk_df[lk_df['type'] == type1]
            for j in range(i + 1, len(ALL_TYPES)):
                type2 = ALL_TYPES[j]
                log_lk_t2 = lk_df[lk_df['type'] == type2]
                n_higher = sum(np.round(log_lk_t1['loglk'] - log_lk_t2['loglk'], 0) > 0)
                log = '{} likelihood\t is higher than {}\t in {:.1f}% of cases'\
                    .format(type1, type2, 100 * n_higher / len(log_lk_t1))
                n_equal = sum(np.round(log_lk_t1['loglk'] - log_lk_t2['loglk'], 0) == 0)
                log += '\n\t is equal to {}\t in {:.1f}% of cases'\
                    .format(type2, 100 * n_equal / len(log_lk_t1))
                n_lower = sum(np.round(log_lk_t1['loglk'] - log_lk_t2['loglk'], 0) < 0)
                log += '\n\t is lower than {}\t in {:.1f}% of cases'\
                    .format(type2, 100 * n_lower / len(log_lk_t1))

                p = binomtest(min(n_higher, n_lower), n_higher + n_lower, 0.5, alternative='two-sided').pvalue
                log += '\n\t is different from {}\t with a p-value of {:g}'.format(type2, p)
                print(log)
                f.write(log + '\n')