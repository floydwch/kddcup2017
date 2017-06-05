# -*- coding: utf-8 -*-
import numpy as np
from sklearn.preprocessing import OneHotEncoder

from lib.feature import backward_targets


def get_one_hot_key():
    encoder = OneHotEncoder(n_values=[3, 2], sparse=False)
    encoder.fit([[0, 0]])

    def one_hot_key(ix, **kargs):
        return encoder.transform([[ix[-2] - 1, ix[-1]]])[0].tolist()

    return one_hot_key


def get_tollgate():
    encoder = OneHotEncoder(n_values=[3], sparse=False)
    encoder.fit([[0]])

    def tollgate(ix, **kargs):
        return encoder.transform([[ix[-2] - 1]])[0].tolist()

    return tollgate


def direction(ix, **kargs):
    return ix[5]


def backward_target_sum(ix, target_df, **kargs):
    return np.nan_to_num(np.nansum(backward_targets(3, ix, target_df)))
