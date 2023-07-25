import numpy as np
import pandas as pd


INTERVAL = 'interval'

INPUT = 'input'


def date2years(d, default_min_date=1900, default_max_date=2017, type=INPUT):
    """
    Converts date to integers provided the dates are month-specific
    :param d: input date to be converted
    :param default_min_date: default min date
    :param default_max_date: default max date
    :param type: interval or input: if interval is specified the min and max dates returns would correspond
        to the 1st day of the date's month and the 1st day of the month following the date's one.
        Otherwise, the middle of the month will be returned.
    :return: date (None if min_date < max_date), min_date, max_date
    """
    if pd.notnull(d):
        first_jan_this_year = pd.datetime(year=d.year, month=1, day=1)
        day_of_this_year = d - first_jan_this_year
        first_jan_next_year = pd.datetime(year=d.year + 1, month=1, day=1)
        days_in_this_year = first_jan_next_year - first_jan_this_year
        min_date = d.year + day_of_this_year / days_in_this_year
        next_month_d = pd.datetime(year=d.year + (0 if d.month < 12 else 1),
                                   month=(d.month if d.month < 12 else 0) + 1, day=1)
        nm_day_of_this_year = (next_month_d - first_jan_this_year).days - 1
        max_date = d.year + nm_day_of_this_year / days_in_this_year.days
        if INTERVAL == type:
            # return None, min_date, max_date
            date = min_date + np.random.random(1)[0] * (max_date - min_date)
            return date, date, date
        else:
            date = (min_date + max_date) / 2
            return date, date, date
    else:
        return None, default_min_date, default_max_date


if '__main__' == __name__:
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--data', default='/home/azhukova/projects/HIV1-UK/data/metadata/metadata.uk.tab', type=str)
    parser.add_argument('--dates', default='/home/azhukova/projects/HIV1-UK/data/metadata/lsd2.interval.dates', type=str)
    parser.add_argument('--date_col', default='sampledate_my', type=str)
    parser.add_argument('--type', required=False, choices=(INTERVAL, INPUT), type=str, default=INTERVAL)
    params = parser.parse_args()

    df = pd.read_csv(params.data, index_col=0, sep='\t')[[params.date_col]]
    df[params.date_col] = pd.to_datetime(df[params.date_col].astype(str).str.replace('.0', '', regex=False),
                                         infer_datetime_format=True)
    m_date = date2years(df[~pd.isna(df[params.date_col])][params.date_col].min())[1]
    M_date = 2017
    df[['date', 'lower', 'upper']] = \
        df[params.date_col].apply(lambda _: date2years(_, m_date, M_date, params.type)).apply(pd.Series)
    df.loc[pd.isna(df['date']), 'date'] = 'b(' + df.loc[pd.isna(df['date']), 'lower'].astype(str) \
                                          + ',' + df.loc[pd.isna(df['date']), 'upper'].astype(str) + ')'
    if params.dates:
        with open(params.dates, 'w+') as f:
            f.write('%d\n' % df.shape[0])
        df['date'].to_csv(params.dates, sep='\t', header=False, mode='a')

