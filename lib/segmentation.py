# -*- coding: utf-8 -*-
from itertools import groupby, product
from functools import partial

import numpy as np
import pandas as pd
import toolz as fn

from .util import dump_to_pickle


def routine(
    dir_path,
    data_name,
    key_cols,
    time_col,
    keys,
    segment_col,
    agg_phase,
    agg,
    other_cols=[],
    mask_holiday=False,
    mode='train'
):
    print('mode: {}'.format(mode))
    print('mask_holiday: {}'.format(mask_holiday))

    train1_df, test1_df, test2_df = map(
        lambda path: pd.read_csv(
            path,
            parse_dates=[time_col],
            usecols=key_cols + [time_col] + other_cols
        ),
        get_df_paths(data_name)
    )

    weather_df = pd.concat(list(map(
        lambda path: pd.read_csv(
            path,
            parse_dates=['date']
        ),
        get_df_paths('weather')
    )))

    sample_df = fn.compose(*filter(fn.identity, (
        partial(
            select_df_by_time,
            lambda ix: (
                backward_time_selector(ix) or
                target_time_selector(ix)
            )
        ),
        partial(agg_df, agg) if agg_phase == 'in segment' else None,
        partial(
            select_df_by_time,
            mask_holiday_selector
        ) if mask_holiday else None,
        partial(segment_df, segment_col)
    )))(pd.concat([train1_df, test1_df, test2_df]))

    train_sample_df, validate_sample_df, test_sample_df = split_df(
        ((10, 18), (10, 25)),
        sample_df
    )

    train_indices = select_observed_indices(train_sample_df)
    validate_indices = (
        select_observed_indices(validate_sample_df)
        if mode == 'predict'
        else make_indices((10, 18), (10, 24), keys)
    )
    test_indices = make_indices((10, 25), (10, 31), keys)

    if agg_phase == 'in segment':
        train_targets = \
            train_sample_df.loc[train_indices].iloc[:, 0].values
        validate_targets = \
            validate_sample_df.loc[validate_indices].iloc[:, 0].values
    elif agg_phase == 'after segment':
        train_targets = agg_df(
            agg,
            train_sample_df.loc[train_indices]
        ).iloc[:, 0].values
        if mode == 'train':
            validate_targets = agg_df(
                np.mean,
                validate_sample_df.loc[validate_indices]
            ).iloc[:, 0].values
        elif mode == 'predict':
            validate_targets = agg_df(
                agg,
                validate_sample_df.loc[validate_indices]
            ).iloc[:, 0].values

    if agg_phase == 'after segment':
        train_target_df = agg_df(agg, train_sample_df)
        validate_target_df = agg_df(agg, validate_sample_df)
        test_target_df = agg_df(agg, test_sample_df)
    else:
        train_target_df = train_sample_df
        validate_target_df = validate_sample_df
        test_target_df = test_sample_df

    output = dict(
        sample_df=sample_df,
        train_target_df=train_target_df,
        validate_target_df=validate_target_df,
        test_target_df=test_target_df,
        train_indices=train_indices,
        validate_indices=validate_indices,
        test_indices=test_indices,
        train_targets=train_targets,
        validate_targets=validate_targets,
        weather_df=weather_df
    )

    dump_to_pickle(dir_path, 'segmentation', output)


def get_df_paths(name):
    return (
        'data/{}/{}.csv'.format(i, name)
        for i in ('train', 'test1', 'test2')
    )


def segment_record(row, key_cols, time_col):
    return tuple(row[k] for k in key_cols) + (
        row[time_col].month,
        row[time_col].day,
        row[time_col].hour,
        row[time_col].minute // 20
    )


def segment_df(segment_col, df):
    def key_fn(record):
        return record[0]

    time_items = sorted(get_time_items(segment_col, df), key=key_fn)
    segmented_items = segment_col_group(time_items, key_fn)

    return pd.DataFrame.from_dict(
        segmented_items, orient='index'
    )


def segment_col_group(time_items, key_fn):
    segmented_col_records = {}
    for time, time_group in groupby(time_items, key_fn):
        col_items = sorted((item[1] for item in time_group), key=key_fn)
        segmented_col_records[time] = {
            col: list(map(lambda x: x[-1], col_group))
            for col, col_group in groupby(col_items, key_fn)
        }
    return segmented_col_records


def get_time_items(segment_col, df):
    for row in df.iterrows():
        for item in segment_col(row):
            yield item


def select_df_by_time(selector, df):
    return df.select(selector)


def train_time_selector(mode, ix):
    month, day, hour = ix[:3]
    if mode == 'train':
        return not(month == 10 and day >= 18)
    elif mode == 'predict':
        return not(month == 10 and day >= 25)


def mask_holiday_selector(ix):
    month, day = ix[:2]
    if month == 9 and 15 <= day <= 17 or \
            month == 10 and 1 <= day <= 7:
        return False
    return True


def target_time_selector(ix):
    month, day, hour = ix[:3]
    if 8 <= hour < 10 or 17 <= hour < 19:
        return True
    return False


def backward_time_selector(ix):
    month, day, hour = ix[:3]
    if 6 <= hour < 8 or 15 <= hour < 17:
        return True
    return False


def select_observed_indices(df):
    return df.iloc[
        df.index.get_level_values(2).isin([8, 9, 17, 18]), :
    ].index.values.tolist()


def make_indices(from_date, to_date, keys):
    return list(
        map(
            lambda x: (*x[0], *x[1:-1], *x[-1]),
            product(
                make_dates(from_date, to_date),
                [8, 9, 17, 18],
                range(3),
                keys
            )
        )
    )


def make_dates(from_date, to_date):
    yield from_date
    new_date = next_date(from_date)
    while new_date <= to_date:
        yield new_date
        new_date = next_date(new_date)


def next_date(date):
    month, day = date
    if month in [1, 3, 5, 7, 8, 10, 12]:
        return increase_date(31, date)
    elif month in [4, 6, 9, 11]:
        return increase_date(30, date)
    elif month == 2:
        return increase_date(29, date)


def increase_date(round_day, date):
    month, day = date
    if day < round_day:
        return month, day + 1
    return month + 1, 1


def split_df(cut_off_dates, df):
    def time_selector(from_date, to_date, ix):
        month, day = ix[:2]
        if from_date is None:
            if (month, day) < to_date:
                return True
        elif from_date <= (month, day) < to_date:
            return True
        return False

    train_df = df.select(partial(time_selector, None, cut_off_dates[0]))
    validate_df = df.select(
        partial(time_selector, cut_off_dates[0], cut_off_dates[1])
    )
    test_df = df.select(
        lambda ix: not(
            time_selector(None, cut_off_dates[1], ix)
        )
    )
    return train_df, validate_df, test_df


def agg_df(agg, df):
    return df.applymap(agg)
