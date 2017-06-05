# -*- coding: utf-8 -*-
import os

import numpy as np

from lib.segmentation import routine, segment_record


def segment_tollgate_by_time(row):
    ix = segment_record(row[1], ['tollgate', 'direction'], 'date_time')
    yield (
        (*ix[-4:], row[1]['tollgate'], row[1]['direction']),
        ('volume', 1)
    )


if __name__ == '__main__':
    routine(
        dir_path=os.path.dirname(__file__),
        data_name='volume',
        key_cols=['tollgate', 'direction'],
        time_col='date_time',
        keys=[(1, 0), (1, 1), (2, 0), (3, 0), (3, 1)],
        segment_col=segment_tollgate_by_time,
        mask_holiday=False,
        agg_phase='in segment',
        agg=np.sum,
        mode=os.environ.get('KDD_MODE', 'train'),
    )
    print('task2 segment done')
