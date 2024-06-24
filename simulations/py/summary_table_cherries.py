import re

import pandas as pd

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Summarize cherry tests.")
    parser.add_argument('--logs', nargs='+', type=str, help="cherry test results")
    parser.add_argument('--tab', type=str, help="summary table")
    params = parser.parse_args()

    df = pd.DataFrame(columns=['model', 'tree', 'PN_test'])

    for log in params.logs:
        i = int(re.findall(r'[0-9]+', log)[-1])
        with open(log, 'r') as f:
            pval = float(f.readline().strip('\n').split('\t')[1])
        model = re.findall(r'BDPN2|BDPN|BDEIPN|BDSSPN|BDEI|BDSS|BD', log)[0]
        df.loc['{}.{}'.format(i, model), :] = [model, i, pval]

    for model in df['model'].unique():
        ddf = df[df['model'] == model]
        mean_pval = ddf['PN_test'].mean()
        min_pval = ddf['PN_test'].min()
        max_pval = ddf['PN_test'].max()
        percentage_significant = 100 * len(ddf[ddf['PN_test'] < 0.05]) / len(ddf)
        print('{}: avg pval {} [{}-{}], {}% have pval < 0.05'
              .format(model, mean_pval, min_pval, max_pval, percentage_significant))
        df.loc['{}.mean'.format(model), :] = [model, 'mean', mean_pval]
        df.loc['{}.min'.format(model), :] = [model, 'min', min_pval]
        df.loc['{}.max'.format(model), :] = [model, 'max', max_pval]
        df.loc['{}.perc'.format(model), :] = [model, 'percent < 0.05', percentage_significant]

    df.sort_values(by=['model', 'tree'], inplace=True)
    df.to_csv(params.tab, sep='\t', index=False)

