
import logging
import re

import pandas as pd

# import glob
# real = glob.glob("/home/azhukova/Evolbioinfo/users/azhukova/projects/bdpn/simulations/medium/BD/tree.*.log")
# est_p = glob.glob("/home/azhukova/Evolbioinfo/users/azhukova/projects/bdpn/simulations/medium/BD/tree.*.p.est_bdpn")
# est_p_bd = glob.glob("/home/azhukova/Evolbioinfo/users/azhukova/projects/bdpn/simulations/medium/BD/tree.*.p.est_bd")
# tab = '/home/azhukova/projects/bdpn/simulations/medium/BD/estimates.tab'

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Summarize errors.")
    parser.add_argument('--estimated_p', nargs='*', default=[], type=str, help="estimated parameters")
    parser.add_argument('--estimated_psi', nargs='*', default=[], type=str, help="estimated parameters")
    parser.add_argument('--estimated_la', nargs='*', default=[], type=str, help="estimated parameters")
    parser.add_argument('--estimated_p_bd', nargs='*', default=[], type=str, help="estimated parameters")
    parser.add_argument('--estimated_psi_bd', nargs='*', default=[], type=str, help="estimated parameters")
    parser.add_argument('--estimated_la_bd', nargs='*', default=[], type=str, help="estimated parameters")
    parser.add_argument('--real', nargs='*', type=str, help="real parameters")
    parser.add_argument('--tab', type=str, help="estimate table")
    params = parser.parse_args()

    logging.getLogger().handlers = []
    logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")

    df = pd.DataFrame(columns=['type', 'sampled_tips',
                               'lambda', 'lambda_min', 'lambda_max',
                               'psi', 'psi_min', 'psi_max',
                               'phi', 'phi_min', 'phi_max',
                               'R_naught', 'R_naught_min', 'R_naught_max',
                               'infectious_time', 'infectious_time_min', 'infectious_time_max',
                               'p', 'p_min', 'p_max',
                               'upsilon', 'upsilon_min', 'upsilon_max',
                               'partner_removal_time', 'partner_removal_time_min', 'partner_removal_time_max'])

    for real in params.real:
        i = int(re.findall(r'[0-9]+', real)[-1])
        ddf = pd.read_csv(real)
        # R0,infectious time,sampling probability,notification probability,removal time after notification,tips,time,hidden_trees
        try:
            R0, it, p, upsilon, rt, tips, T, h_trees \
                = ddf.loc[next(iter(ddf.index)), :]
            df.loc['{}.real'.format(i),
                   ['R_naught', 'infectious_time', 'partner_removal_time',
                    'lambda', 'psi', 'phi', 'p', 'upsilon', 'sampled_tips', 'type']] \
                = [R0, it, rt, R0 / it, 1 / it, 1 / rt, p, upsilon, tips, 'real']
        except:
            # This is a BD model tree actually
            R0, it, p, tips, T, h_trees \
                = ddf.loc[next(iter(ddf.index)), :]
            df.loc['{}.real'.format(i),
                   ['R_naught', 'infectious_time',
                    'lambda', 'psi', 'p', 'sampled_tips', 'type']] \
                = [R0, it, R0 / it, 1 / it, p, tips, 'real']

    estimate_list = []
    if params.estimated_la:
        estimate_list.append((params.estimated_la, 'lambda'))
    if params.estimated_psi:
        estimate_list.append((params.estimated_psi, 'psi'))
    if params.estimated_p:
        estimate_list.append((params.estimated_p, 'p'))

    for (est_list, fixed) in estimate_list:
        if not est_list:
            continue
        for est in est_list:
            i = int(re.findall(r'[0-9]+', est)[0])
            ddf = pd.read_csv(est, index_col=0)
            est_label = 'BDPN({})'.format(fixed)
            R0, rt, rho, upsilon, prt, la, psi, phi = ddf.loc['value', :]
            df.loc['{}.{}'.format(i, est_label),
            ['R_naught', 'infectious_time', 'partner_removal_time',
             'lambda', 'psi', 'phi', 'p', 'upsilon', 'type']] \
                = [R0, rt, prt, la, psi, phi, rho, upsilon, est_label]
            if 'CI_min' in ddf.index:
                R0, rt, rho, upsilon, prt, la, psi, phi = ddf.loc['CI_min', :]
                df.loc['{}.{}'.format(i, est_label),
                ['R_naught_min', 'infectious_time_min', 'partner_removal_time_min',
                 'lambda_min', 'psi_min', 'phi_min', 'p_min', 'upsilon_min', 'type']] \
                    = [R0, rt, prt, la, psi, phi, rho, upsilon, est_label]
                R0, rt, rho, upsilon, prt, la, psi, phi = ddf.loc['CI_max', :]
                df.loc['{}.{}'.format(i, est_label),
                ['R_naught_max', 'infectious_time_max', 'partner_removal_time_max',
                 'lambda_max', 'psi_max', 'phi_max', 'p_max', 'upsilon_max', 'type']] \
                    = [R0, rt, prt, la, psi, phi, rho, upsilon, est_label]

    estimate_list = []
    if params.estimated_la_bd:
        estimate_list.append((params.estimated_la_bd, 'lambda'))
    if params.estimated_psi_bd:
        estimate_list.append((params.estimated_psi_bd, 'psi'))
    if params.estimated_p_bd:
        estimate_list.append((params.estimated_p_bd, 'p'))

    for (est_list, fixed) in estimate_list:
        if not est_list:
            continue
        for est in est_list:
            i = int(re.findall(r'[0-9]+', est)[0])
            ddf = pd.read_csv(est, index_col=0)
            est_label = 'BD({})'.format(fixed)
            R0, rt, rho, la, psi = ddf.loc['value', :]
            df.loc['{}.{}'.format(i, est_label),
            ['R_naught', 'infectious_time',
             'lambda', 'psi', 'p', 'type']] \
                = [R0, rt, la, psi, rho, est_label]
            if 'CI_min' in ddf.index:
                R0, rt, rho, la, psi = ddf.loc['CI_min', :]
                df.loc['{}.{}'.format(i, est_label),
                ['R_naught_min', 'infectious_time_min',
                 'lambda_min', 'psi_min', 'p_min', 'type']] \
                    = [R0, rt, la, psi, rho, est_label]
                R0, rt, rho, la, psi = ddf.loc['CI_max', :]
                df.loc['{}.{}'.format(i, est_label),
                ['R_naught_max', 'infectious_time_max',
                 'lambda_max', 'psi_max', 'p_max', 'type']] \
                    = [R0, rt, la, psi, rho, est_label]

    df.index = df.index.map(lambda _: int(_.split('.')[0]))
    df.sort_index(inplace=True)
    df.to_csv(params.tab, sep='\t')
