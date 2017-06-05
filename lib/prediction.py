# -*- coding: utf-8 -*-
import os
from datetime import datetime

import pandas as pd

from .util import load_from_pickle


def routine(
    dir_path, task_id, make_row, cols, mode='train'
):
    model, test_indices, test_features = get_data(dir_path, mode)
    preds = model.predict(test_features)

    df = pd.DataFrame(
        [make_row(ix, pred) for ix, pred in zip(test_indices, preds)],
        columns=cols
    )

    submission_dir = os.path.join(dir_path, 'submission')
    if not os.path.exists(submission_dir):
        os.mkdir(submission_dir)

    time = datetime.now().strftime('%d-%H-%M')

    df.to_csv(
        os.path.join(
            submission_dir,
            'task{}-{}.csv'
            .format(task_id, time)
        ),
        index=False
    )


def get_data(path, mode):
    segment_data = load_from_pickle(path, 'segmentation')
    feature_data = load_from_pickle(path, 'feature')

    if mode == 'train':
        test_indices = segment_data['validate_indices']
        test_features = feature_data['validate_features']
    elif mode == 'predict':
        test_indices = segment_data['test_indices']
        test_features = feature_data['test_features']

    return (
        load_from_pickle(path, 'model')['model'],
        test_indices,
        test_features
    )


def make_time_window(month, day, hour, window):
    end_hour = hour if window < 2 else hour + 1
    minute = window % 3 * 20

    start_time = '{:02}:{:02}:00'.format(hour, minute)
    end_time = '{:02}:{:02}:00'.format(end_hour, (minute + 20) % 60)

    return '[2016-{}-{} {},2016-{}-{} {})'.format(
        month, day, start_time, month, day, end_time
    )
