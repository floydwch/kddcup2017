# -*- coding: utf-8 -*-
from collections import Counter
from itertools import chain

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction import DictVectorizer

from lib.extraction import to_list


def get_one_hot_key():
    encoder = OneHotEncoder(n_values=[3, 3], sparse=False)
    encoder.fit([[0, 0]])
    intersection_id_map = dict(
        A=0,
        B=1,
        C=2
    )

    def one_hot_key(ix, **kargs):
        return encoder.transform([
            [
                intersection_id_map[ix[-2]],
                ix[-1] - 1
            ]
        ])[0].tolist()

    return one_hot_key


def get_route_features():
    route_df = pd.read_csv('data/train/routes.csv')
    link_df = pd.read_csv('data/train/links.csv')

    route_encoder = DictVectorizer(sparse=False)
    route_encoder.fit([Counter(link_df['link_id'])])

    def get_link_df(route):
        return link_df.query('link_id in {}'.format(
            list(map(int, route[1]['link_seq'].split(',')))
        ))

    def one_hot(route):
        return route_encoder.transform(
            [Counter(map(int, route[1]['link_seq'].split(',')))]
        )[0].tolist()

    def length(route):
        return get_link_df(route)['length'].sum()

    def volume(route):
        return get_link_df(route).apply(
            lambda row: row['width'] * row['length'],
            1
        ).sum()

    def max_link_width(route):
        return get_link_df(route)['width'].max()

    def min_link_width(route):
        return get_link_df(route)['width'].min()

    def mean_link_width(route):
        return get_link_df(route)['width'].mean()

    def median_link_width(route):
        return get_link_df(route)['width'].median()

    feature_fns = [
        # one_hot,
        length,
        volume,
        # max_link_width,
        # min_link_width,
        mean_link_width,
        # median_link_width
    ]

    route_features_map = {
        (route[1]['intersection_id'], route[1]['tollgate_id']):
            list(chain.from_iterable(
                [to_list(fn(route)) for fn in feature_fns]
            ))
        for route in route_df.iterrows()
    }

    def route_features(ix, **kargs):
        return route_features_map[ix[-2], ix[-1]]

    return route_features
