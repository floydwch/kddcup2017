# -*- coding: utf-8 -*-
import os
import logging

import numpy as np
from sklearn.model_selection import (
    cross_val_score, KFold, RandomizedSearchCV, GridSearchCV
)
from sklearn.metrics import make_scorer

from .util import dump_to_pickle, load_from_pickle
from .metric import abs_per_err, mean_abs_per_err
from .extraction import feature_df_to_array, get_selected_feature_names


def routine(
    dir_path,
    task_id,
    model, hyperparam_dists=None, searcher_type='random',
    masked_features=[],
    mode='train'
):
    logging.basicConfig(
        filename='task{}-train.log'.format(task_id),
        level=logging.INFO,
        format='%(message)s'
    )

    print('mode: {}'.format(mode))

    (
        train_features, train_targets,
        validate_features, validate_targets,
        test_features,
    ) = get_data(dir_path, masked_features)

    print('n_features: {}'.format(train_features.shape[1]))
    print('n_samples: {}'.format(train_features.shape[0]))

    masked_train_indices = \
        ~np.sum(np.isnan(train_features), 1).astype(bool)

    print('n_masked_train_indices: {}'.format(masked_train_indices.sum()))

    scorer = make_scorer(mean_abs_per_err, False)

    if hyperparam_dists:
        cv_errs = None

        print('hyperparams searching...')

        if searcher_type == 'random':
            searcher = RandomizedSearchCV(
                model,
                param_distributions=hyperparam_dists,
                cv=KFold(5, True, 0),
                n_iter=100,
                n_jobs=-1,
                scoring=scorer,
                random_state=0
            )
        elif searcher_type == 'grid':
            searcher = GridSearchCV(
                model,
                param_grid=hyperparam_dists,
                cv=KFold(5, True, 0),
                n_jobs=-1,
                scoring=scorer
            )

        searcher.fit(
            train_features[masked_train_indices],
            train_targets[masked_train_indices]
        )

        log_info('hyperparams: {}'.format(searcher.best_params_))

        model = searcher.best_estimator_
    else:
        cv_errs = -1 * cross_val_score(
            model,
            train_features[masked_train_indices],
            train_targets[masked_train_indices],
            cv=KFold(4, True, 0),
            scoring=scorer
        )

        log_info('hyperparams: {}'.format(model.get_params()))

        model.fit(
            train_features[masked_train_indices],
            train_targets[masked_train_indices]
        )

    train_predictions = \
        model.predict(train_features[masked_train_indices])
    validate_predictions = model.predict(validate_features)

    train_err = mean_abs_per_err(
        train_targets[masked_train_indices], train_predictions
    )
    validate_err = mean_abs_per_err(
        validate_targets, validate_predictions
    )
    validate_std = \
        np.std(abs_per_err(validate_targets, validate_predictions))
    max_validate_err = np.max(
        abs_per_err(validate_targets, validate_predictions)
    )
    min_validate_err = np.min(
        abs_per_err(validate_targets, validate_predictions)
    )

    report(
        cv_errs,
        train_err, validate_err,
        validate_std,
        max_validate_err,
        min_validate_err
    )

    if mode == 'predict':
        all_train_features = \
            np.vstack([
                train_features[masked_train_indices],
                validate_features
            ])
        all_train_targets = np.hstack([
            train_targets[masked_train_indices],
            validate_targets
        ])
        model.fit(all_train_features, all_train_targets)

    dump_to_pickle(dir_path, 'model', dict(model=model))


def get_data(path, masked_features):
    feature_data = load_from_pickle(path, 'extraction')
    train_feature_df = feature_data['train_feature_df']
    validate_feature_df = feature_data['validate_feature_df']
    test_feature_df = feature_data['test_feature_df']

    segment_data = load_from_pickle(path, 'segmentation')
    train_targets = segment_data['train_targets']
    validate_targets = segment_data['validate_targets']

    if os.path.isfile('feature.pickle'):
        feature_data = load_from_pickle(path, 'feature')
        train_features = feature_data['train_features']
        validate_features = feature_data['validate_features']
        test_features = feature_data['test_features']
    else:
        selected_feature_names = get_selected_feature_names(
            train_feature_df, masked_features
        )

        train_features, validate_features, test_features = map(
            lambda df: (
                feature_df_to_array(df, selected_feature_names)
            ),
            (train_feature_df, validate_feature_df, test_feature_df)
        )

        dump_to_pickle(path, 'feature', dict(
            train_features=train_features,
            validate_features=validate_features,
            test_features=test_features
        ))

    return (
        train_features, train_targets,
        validate_features, validate_targets,
        test_features
    )


def report(
    cv_errs,
    train_err, validate_err,
    validate_std,
    max_validate_err,
    min_validate_err
):
    if cv_errs is not None:
        log_info('cv errors: {}'.format(cv_errs))
        log_info('cv errors mean: {:.4f}'.format(cv_errs.mean()))
    log_info('training error: {:.4f}'.format(train_err))
    log_info('validation error: {:.4f}'.format(validate_err))
    log_info('validation std: {}'.format(validate_std))
    log_info('max validation error: {}'.format(max_validate_err))
    log_info('min validation error: {}'.format(min_validate_err))


def log_info(message):
    logging.info(message)
    print(message)
