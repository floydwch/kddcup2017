# -*- coding: utf-8 -*-
import os

from lib.prediction import make_time_window, routine


def make_row(ix, pred):
    tollgate_id, direction = ix[-2:]
    return tollgate_id, make_time_window(*ix[:4]), direction, pred


if __name__ == '__main__':
    routine(
        dir_path=os.path.dirname(__file__),
        task_id=2,
        make_row=make_row,
        cols=[
            'tollgate_id',
            'time_window',
            'direction',
            'volume'
        ],
        mode=os.environ.get('KDD_MODE', 'train')
    )
    print('task2 predict done')
