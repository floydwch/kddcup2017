# -*- coding: utf-8 -*-
import os
from itertools import starmap, chain
from functools import reduce, partial, update_wrapper

import numpy as np
from pandas import DataFrame, Series

from .util import dump_to_pickle, load_from_pickle


def routine(dir_path, feature_fns, ext_feature_fns=[]):
    (
        sample_df,
        train_indices,
        validate_indices,
        test_indices,
        train_target_df,
        validate_target_df,
        test_target_df,
        weather_df
    ) = get_data(dir_path)

    train_feature_df, validate_feature_df, test_feature_df = starmap(
        lambda indices, sample_df, target_df: DataFrame.from_dict(
            dict(map(
                lambda fn: (
                    fn.__name__, list(map(
                        lambda ix: fn(
                            ix=ix,
                            sample_df=sample_df,
                            target_df=target_df,
                            weather_df=weather_df
                        ),
                        indices
                    ))
                ),
                feature_fns
            ))
        ),
        (
            (train_indices, sample_df, train_target_df),
            (validate_indices, sample_df, validate_target_df),
            (test_indices, sample_df, test_target_df)
        )
    )

    ext_train_feature_df, ext_validate_feature_df, ext_test_feature_df = \
        map(
            lambda feature_df: reduce(
                lambda feature_df, fn: feature_df.assign(
                    **{fn.__name__: fn(feature_df)}
                ),
                ext_feature_fns,
                feature_df
            ),
            (train_feature_df, validate_feature_df, test_feature_df)
        )

    output = dict(
        train_feature_df=ext_train_feature_df,
        validate_feature_df=ext_validate_feature_df,
        test_feature_df=ext_test_feature_df
    )

    dump_to_pickle(dir_path, 'extraction', output)

    if os.path.isfile('feature.pickle'):
        os.remove(dir_path, 'feature.pickle')
    if os.path.isfile('model.pickle'):
        os.remove(dir_path, 'model.pickle')


def get_data(path):
    data = load_from_pickle(path, 'segmentation')
    return (
        data['sample_df'],
        data['train_indices'],
        data['validate_indices'],
        data['test_indices'],
        data['train_target_df'],
        data['validate_target_df'],
        data['test_target_df'],
        data['weather_df']
    )


def get_selected_feature_names(feature_df, masked_features=[]):
    return list(
        set(list(feature_df)) -
        set((
            fn if isinstance(fn, str) else fn.__name__
            for fn in masked_features
        ))
    )


def feature_df_to_array(df, selected_feature_names):
    return np.array(
        df[selected_feature_names].apply(
            lambda row: Series(
                list(chain.from_iterable(map(to_list, row)))
            ),
            1
        ).values.astype(float)
    )


def to_list(value):
    return value if isinstance(value, list) else [value]


def wrapped_partial(fn, *args, **kwargs):
    partial_fn = partial(fn, *args, **kwargs)
    update_wrapper(partial_fn, fn)
    return partial_fn
