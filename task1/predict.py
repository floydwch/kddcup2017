# -*- coding: utf-8 -*-
import os

from lib.prediction import make_time_window, routine


def make_row(ix, pred):
    intersection_id, tollgate_id = ix[-2:]
    return intersection_id, tollgate_id, make_time_window(*ix[:4]), pred


if __name__ == '__main__':
    routine(
        dir_path=os.path.dirname(__file__),
        task_id=1,
        make_row=make_row,
        cols=[
            'intersection_id',
            'tollgate_id',
            'time_window',
            'avg_travel_time'
        ],
        mode=os.environ.get('KDD_MODE', 'train')
    )
    print('task1 predict done')
