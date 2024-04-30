from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.offsetbox import TextArea, HPacker, AnchoredOffsetbox, VPacker
from statsmodels.stats.weightstats import CompareMeans

RATE_PARAMETERS = ['lambda', 'psi', 'p']
EPIDEMIOLOGIC_PARAMETERS = ['R_naught', 'infectious_time']
par2greek = {'lambda': u'\u03bb', 'psi': u'\u03c8', 'p': '\u03c1',
             'R_naught': u'\u0052\u2080' + '=' + u'\u03bb\u002F\u03c8',
             'infectious_time': 'infectious time 1' + u'\u002F\u03c8'}
PARAMETERS = RATE_PARAMETERS + EPIDEMIOLOGIC_PARAMETERS

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plots errors.")
    parser.add_argument('--estimates', type=str, help="estimated parameters")
    parser.add_argument('--pdf', type=str, help="plot")
    parser.add_argument('--tab', type=str, help="error table")
    parser.add_argument('--fixed', type=str, default='p', help="fixed parameter", choices=RATE_PARAMETERS)
    params = parser.parse_args()

    df = pd.read_csv(params.estimates, sep='\t', index_col=0)

    real_df = df.loc[df['type'] == 'real', :]

    df = df.loc[df['type'] != 'real', :]
    df = df.loc[real_df.index, :]
    types = sorted(df['type'].unique(), key=lambda _: next(i for (i, par) in enumerate(RATE_PARAMETERS) if par in _))
    types = [_ for _ in types if '({})'.format(params.fixed) in _]

    for type in types:
        mask = df['type'] == type
        for par in PARAMETERS:
            # df.loc[mask, '{}_error'.format(par)] = (df.loc[mask, par] - real_df[par]) / real_df[par]
            if par != 'p':
                df.loc[mask, '{}_error'.format(par)] = (df.loc[mask, par] - real_df[par]) / real_df[par]
            else:
                df.loc[mask, '{}_error'.format(par)] = (df.loc[mask, par] - real_df[par])

    error_columns = [col for col in df.columns if 'error' in col]
    df[['type'] + PARAMETERS + error_columns].to_csv(params.tab, sep='\t')

    plt.clf()
    n_types = len(types)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 12))
    rc = {'font.size': 12, 'axes.labelsize': 10, 'legend.fontsize': 10, 'axes.titlesize': 10, 'xtick.labelsize': 10,
          'ytick.labelsize': 10}
    sns.set(style="whitegrid")
    sns.set(rc=rc)

    abs_error_or_1 = lambda _: min(abs(_), 1)

    RATE_PARAMETERS = [_ for _ in RATE_PARAMETERS if _ != params.fixed]
    for pars, ax in ((RATE_PARAMETERS, ax1), (EPIDEMIOLOGIC_PARAMETERS, ax2)):
        data = []
        par2type2avg_error = defaultdict(lambda: dict())
        for type in types:
            type_label = type
            for _ in RATE_PARAMETERS:
                if "({})".format(_) in type:
                    type_label = par2greek[_] + ' fixed ({})'.format('BDPN' if 'BDPN' in type else 'BD')

            for par in pars:
                data.extend([[par2greek[par], _, type_label]
                             for _ in df.loc[df['type'] == type, '{}_error'.format(par)].apply(abs_error_or_1)])
                if '({})'.format(par) not in type and '({})'.format(par[:2]) not in type \
                        and (par != 'partner_removal_time' or '(psi_p)' not in type) \
                        and (par != 'infectious_time' or '(psi)' not in type):
                    par2type2avg_error[par][type] = \
                        '{:.2f}({:.2f})'.format(np.mean(np.abs(df.loc[df['type'] == type, '{}_error'.format(par)])),
                                                np.mean(df.loc[df['type'] == type, '{}_error'.format(par)]))
                else:
                    par2type2avg_error[par][type] = '(fixed)   '

        par2types2pval = defaultdict(lambda: dict())
        for par in pars:
            for i in range(n_types):
                type_1 = types[i]
                for j in range(i + 1, n_types):
                    type_2 = types[j]
                    pval_abs = \
                        CompareMeans.from_data(data1=df.loc[df['type'] == type_1, '{}_error'.format(par)].apply(np.abs),
                                               data2=df.loc[df['type'] == type_2, '{}_error'.format(par)].apply(np.abs)).ztest_ind()[1]
                    if 'forest' in type_1 or 'forest' in type_2:
                        pval_abs = 1
                    par2types2pval[par][(type_1, type_2)] = pval_abs

        ERROR_COL = 'relative error' if 'p' not in pars else 'relative or absolute (for {} and {}) error'.format(par2greek['p'], par2greek['pn'])
        plot_df = pd.DataFrame(data=data, columns=['parameter', ERROR_COL, 'config'])

        palette = sns.color_palette("colorblind")
        ax = sns.swarmplot(x="parameter", y=ERROR_COL, palette=palette, data=plot_df, alpha=.8, hue="config", ax=ax,
                           dodge=True)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        min_error = min(min(df['{}_error'.format(_)]) for _ in pars)
        max_error = max(max(df['{}_error'.format(_)]) for _ in pars)
        abs_error = max(max_error, abs(min_error))
        ax.set_yticks(list(np.arange(0, min(1.1, abs_error + 0.1), step=0.2 if abs_error >= 1 else 0.1)))
        if abs_error >= 1:
            ax.set_yticklabels(['{:.1f}'.format(_) for _ in np.arange(0, 1.0, step=0.2)] + [u"\u22651"])
        ax.set_ylim(0, min(1.1, abs_error + 0.1))
        ax.yaxis.grid()

        def get_xbox(par):
            boxes = [TextArea(text, textprops=dict(color=color, ha='center', va='center', fontsize='x-small',
                                                   fontweight='bold'))
                     for text, color in zip((par2type2avg_error[par][_] for _ in types), palette)]
            return HPacker(children=boxes, align="center", pad=0, sep=4 if len(pars) == 3 else 0)
        xbox = HPacker(children=[get_xbox(par) for par in pars], align="center", pad=0, sep=120)
        anchored_xbox = AnchoredOffsetbox(loc=3, child=xbox, pad=0, frameon=False,
                                          bbox_to_anchor=(0.2, -0.08),
                                          bbox_transform=ax.transAxes, borderpad=0.)
        ax.set_xlabel('')

        ax.add_artist(anchored_xbox)
        # ax.add_artist(anchored_xbox)

        ax.set_xlabel('')
        leg = ax.legend()
        # if pars != RATE_PARAMETERS:
        #     leg.remove()

    plt.tight_layout()
    fig.set_size_inches(6, 9)
    # plt.show()
    plt.savefig(params.pdf, dpi=300)
