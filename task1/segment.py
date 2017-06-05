# -*- coding: utf-8 -*-
import os

import numpy as np

from lib.segmentation import routine, segment_record


def segment_route_by_time(row):
    ix = segment_record(
        row[1],
        ['intersection_id', 'tollgate_id'],
        'starting_time'
    )

    yield (
        (*ix[-4:], row[1]['intersection_id'], row[1]['tollgate_id']),
        ('travel_time', row[1]['travel_time'])
    )


if __name__ == '__main__':
    routine(
        dir_path=os.path.dirname(__file__),
        data_name='trajectories',
        key_cols=['intersection_id', 'tollgate_id'],
        time_col='starting_time',
        other_cols=['travel_time'],
        keys=[('A', 2), ('A', 3), ('B', 1), ('B', 3), ('C', 1), ('C', 3)],
        segment_col=segment_route_by_time,
        mask_holiday=False,
        agg_phase='after segment',
        agg=np.mean,
        mode=os.environ.get('KDD_MODE', 'train')
    )
    print('task1 segment done')
