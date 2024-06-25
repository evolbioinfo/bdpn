import os

import numpy as np
import pandas as pd

RATE_PARAMETERS = ['lambda', 'psi', 'phi']
PARAMETERS = RATE_PARAMETERS + ['p', 'upsilon']


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Access CIs.")
    parser.add_argument('--estimates', type=str, help="estimated parameters",
                        default=os.path.join(os.path.dirname(__file__), '..', 'trees', 'BDPN', 'estimates.tab'))
    parser.add_argument('--log', type=str, help="output log",
                        default=os.path.join(os.path.dirname(__file__), '..', 'trees', 'BDPN', 'CIs.log'))
    params = parser.parse_args()

    df = pd.read_csv(params.estimates, sep='\t', index_col=0)

    real_df = df.loc[df['type'] == 'real', :]
    df = df.loc[df['type'] != 'real', :]
    with open(params.log, 'w+') as f:
        for type in df['type'].unique():
            mask = df['type'] == type
            print('\n================{}==============='.format(type))
            f.write('\n================{}===============\n'.format(type))
            n_observations = sum(mask)
            for par in PARAMETERS:
                df.loc[mask, '{}_within_CI'.format(par)] \
                    = (np.less_equal(df.loc[mask, '{}_min'.format(par)], real_df[par])
                       | np.less_equal(np.abs(real_df[par] - df.loc[mask, '{}_min'.format(par)]), 1e-3)) \
                      & (np.less_equal(real_df[par], df.loc[mask, '{}_max'.format(par)])
                         | np.less_equal(np.abs(df.loc[mask, '{}_max'.format(par)] - real_df[par]), 1e-3))
                print('{}:\t{:.1f}% within CIs'
                      .format(par, 100 * sum(df.loc[mask, '{}_within_CI'.format(par)]) / n_observations))
                f.write('{}:\t{:.1f}% within CIs\n'
                      .format(par, 100 * sum(df.loc[mask, '{}_within_CI'.format(par)]) / n_observations))
                df.loc[mask, '{}_CI_relative_width'.format(par)] \
                    = 100 * (df.loc[mask, '{}_max'.format(par)] - df.loc[mask, '{}_min'.format(par)]) / real_df[par]
                print('{}:\t{:.1f}% median CI width'
                      .format(par, (df.loc[mask, '{}_CI_relative_width'.format(par)].median())))
                f.write('{}:\t{:.1f}% median CI width\n'
                      .format(par, (df.loc[mask, '{}_CI_relative_width'.format(par)].median())))