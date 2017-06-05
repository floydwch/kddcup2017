# -*- coding: utf-8 -*-
from operator import mul
from itertools import starmap

import numpy as np
from sklearn.base import BaseEstimator

from .metric import mean_abs_per_err


class Model(BaseEstimator):
    def __init__(self, estimators, weights):
        self.estimators = estimators
        self.weights = weights
        assert .99999 <= sum(weights) <= 1

    def fit(self, train_features, train_targets, *args):
        for estimator in self.estimators:
            estimator.fit(
                train_features,
                train_targets,
                *args
            )

    def predict(self, features):
        preds = [
            estimator.predict(features)
            for i, estimator in enumerate(self.estimators)
        ]
        return np.sum(list(starmap(mul, zip(self.weights, preds))), 0)

    def score(self, features, targets):
        return 1 - mean_abs_per_err(
            targets, self.predict(features)
        )
