from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.offsetbox import TextArea, HPacker, AnchoredOffsetbox, VPacker
from statsmodels.stats.weightstats import CompareMeans

RATE_PARAMETERS = ['lambda', 'psi', 'psi_p', 'p', 'pn']
EPIDEMIOLOGIC_PARAMETERS = ['R_naught', 'infectious_time', 'partner_removal_time']
par2greek = {'lambda': u'\u03bb', 'psi': u'\u03c8', 'psi_p': u'\u03c8p', 'p': '\u03c1', 'pn': '\u03c5',
             'R_naught': u'\u0052\u2080' + '=' + u'\u03bb\u002F\u03c8',
             'infectious_time': 'infectious time 1' + u'\u002F\u03c8',
             'partner_removal_time': 'partner removal time 1' + u'\u002F\u03c6'}
PARAMETERS = RATE_PARAMETERS + EPIDEMIOLOGIC_PARAMETERS

PS_BDPN = EPIDEMIOLOGIC_PARAMETERS + ['p', 'pn']
PS_BD = EPIDEMIOLOGIC_PARAMETERS[:-1] + ['p']

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plots errors.")
    parser.add_argument('--estimates_bdpn', type=str, help="estimated parameters")
    parser.add_argument('--estimates_bd', type=str, help="estimated parameters")
    parser.add_argument('--pdf', type=str, help="plot")
    parser.add_argument('--fixed', type=str, default='p', help="fixed parameter", choices=RATE_PARAMETERS)
    # parser.add_argument('--pn_min', type=float, default=0., help="Only display cases with pn greater or equal to this value")
    # parser.add_argument('--pn_max', type=float, default=1, help="Only display cases with pn smaller than this value")
    params = parser.parse_args()

    plt.clf()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    rc = {'font.size': 12, 'axes.labelsize': 10, 'legend.fontsize': 10, 'axes.titlesize': 10, 'xtick.labelsize': 10,
          'ytick.labelsize': 10}
    sns.set(style="whitegrid")
    sns.set(rc=rc)

    for est, ax, pars in ((params.estimates_bdpn, ax1, PS_BDPN), (params.estimates_bd, ax2, PS_BD)):
        df = pd.read_csv(est, sep='\t', index_col=0)

        real_df = df.loc[df['type'] == 'real', :]
        # real_df = real_df[(real_df['pn'] * real_df['p'] >= params.pn_min) & (real_df['pn'] * real_df['p'] < params.pn_max)]

        df = df.loc[df['type'] != 'real', :]
        df = df.loc[real_df.index, :]
        types = sorted(df['type'].unique(), key=lambda _: next(i for (i, par) in enumerate(RATE_PARAMETERS) if par in _))
        types = [_ for _ in types if '({})'.format(params.fixed) in _]

        for type in types:
            mask = df['type'] == type
            for par in PARAMETERS:
                # df.loc[mask, '{}_error'.format(par)] = (df.loc[mask, par] - real_df[par]) / real_df[par]
                if par != 'p' and par != 'pn':
                    if 'PN' in type or par != 'psi_p':
                        df.loc[mask, '{}_error'.format(par)] = (df.loc[mask, par] - real_df[par]) / real_df[par]
                else:
                    if 'PN' in type or par != 'pn':
                        df.loc[mask, '{}_error'.format(par)] = (df.loc[mask, par] - real_df[par])


        n_types = len(types)

        abs_error_or_1 = lambda _: min(abs(_), 1)

        data = []
        par2type2avg_error = defaultdict(lambda: dict())

        if params.fixed in pars:
            pars.remove(params.fixed)
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
        xbox = HPacker(children=[get_xbox(par) for par in pars], align="center", pad=0, sep=65 if len(pars) == 3 else 16)
        anchored_xbox = AnchoredOffsetbox(loc=3, child=xbox, pad=0, frameon=False,
                                          bbox_to_anchor=(0.07 if len(pars) == 3 else 0.04, -0.08),
                                          bbox_transform=ax.transAxes, borderpad=0.)
        ax.set_xlabel('')

        ax.add_artist(anchored_xbox)

        def get_pbox(par):
            EMPTY = ' ' * 10
            LONG_DASH = u"\u2014"
            FILLED = LONG_DASH * 10
            boxes = []
            for i in range(n_types - 1):
                type_1 = types[i]
                s = EMPTY * max(0, i)
                skip = par in type_1 or (par == 'infectious_time' and 'psi' in type_1) \
                       or (par == 'incubation_period' and 'mu' in type_1)
                for j in range(i + 1, n_types):
                    type_2 = types[j]
                    skip = skip or par in type_2 or (par == 'infectious_time' and 'psi' in type_2) \
                       or (par == 'incubation_period' and 'mu' in type_2)
                    pval = par2types2pval[par][(type_1, type_2)]
                    if not skip:
                        print(par, type_1, type_2, pval)
                    if not skip and pval < 0.05:
                        boxes.append(TextArea(s + LONG_DASH * 3 + '{:.5f}'.format(pval) + LONG_DASH * 3 + EMPTY * (n_types - j - 1),
                                              textprops=dict(color='black', ha='center', va='center',
                                                             fontsize='x-small', fontweight='bold', family='monospace')))
                    else:
                        boxes.append(TextArea(EMPTY * n_types,
                                              textprops=dict(color='black', ha='center', va='center',
                                                             fontsize='x-small', fontweight='bold', family='monospace')))
                    s += FILLED
            return VPacker(children=list(reversed(boxes)), mode='equal', pad=0, sep=3) if len(boxes) > 1 else boxes[0]

        xbox = HPacker(children=[get_pbox(par) for par in pars], align="center", pad=0, sep=20)
        anchored_xbox = AnchoredOffsetbox(loc=3, child=xbox, pad=0, frameon=False,
                                          bbox_to_anchor=(0.17 if n_types == 3 else 0.20, 1),
                                          bbox_transform=ax.transAxes, borderpad=0.)
        ax.add_artist(anchored_xbox)

        ax.set_xlabel('')
        leg = ax.legend()
        # if pars != RATE_PARAMETERS:
        #     leg.remove()

    plt.tight_layout()
    fig.set_size_inches(12, 9)
    # plt.show()
    plt.savefig(params.pdf, dpi=300)
