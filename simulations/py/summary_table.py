import logging
import re

import pandas as pd


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Summarize errors.")
    parser.add_argument('--estimated_p', nargs='+', type=str, help="estimated parameters")
    parser.add_argument('--estimated_psi', nargs='+', type=str, help="estimated parameters")
    parser.add_argument('--estimated_la', nargs='+', type=str, help="estimated parameters")
    parser.add_argument('--real', nargs='+', type=str, help="real parameters")
    parser.add_argument('--tab', type=str, help="estimate table")
    params = parser.parse_args()

    logging.getLogger().handlers = []
    logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")

    df = pd.DataFrame(columns=['type', 'sampled_tips',
                               'lambda',
                               'psi',
                               'psi_p',
                               'R_naught',
                               'infectious_time',
                               'p',
                               'pn',
                               'partner_removal_time'])

    for real in params.real:
        i = int(re.findall(r'[0-9]+', real)[0])
        ddf = pd.read_csv(real)
        # R0,infectious time,sampling probability,notification probability,removal time after notification,tips,time,hidden_trees
        R0, it, p, pn, rt, tips, T, h_trees \
            = ddf.loc[next(iter(ddf.index)), :]
        df.loc['{}.real'.format(i),
               ['R_naught', 'infectious_time', 'partner_removal_time',
                'lambda', 'psi', 'psi_p', 'p', 'pn', 'sampled_tips', 'type']] \
            = [R0, it, rt, R0 / it, 1 / it, 1 / rt, p, pn, tips, 'real']

    for (est_list, fixed) in ((params.estimated_la, 'lambda'), (params.estimated_psi, 'psi'), (params.estimated_p, 'p')):
        for est in est_list:
            i = int(re.findall(r'[0-9]+', est)[0])
            ddf = pd.read_csv(est)
            est_label = 'BDPN({})'.format(fixed)
            estimates = ddf.loc[next(iter(ddf.index)), :]
            R0, it, p, pn, rt = ddf.loc[next(iter(ddf.index)), :]
            df.loc['{}.{}'.format(i, est_label),
            ['R_naught', 'infectious_time', 'partner_removal_time',
             'lambda', 'psi', 'psi_p', 'p', 'pn', 'type']] \
                = [R0, it, rt, R0 / it, 1 / it, 1 / rt, p, pn, est_label]

    df.index = df.index.map(lambda _: int(_.split('.')[0]))
    df.sort_index(inplace=True)
    df[['type', 'sampled_tips',
        'lambda',
        'psi',
        'psi_p',
        'p',
        'pn',
        'R_naught',
        'infectious_time',
        'partner_removal_time']].to_csv(params.tab, sep='\t')
