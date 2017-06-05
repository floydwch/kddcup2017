# -*- coding: utf-8 -*-
import os

from lib.extraction import routine, wrapped_partial
from lib.feature import (
    off_work, is_working_day,
    time_window,
    backward_target_median
)

from .feature import get_one_hot_key, get_route_features


if __name__ == '__main__':
    routine(dir_path=os.path.dirname(__file__), feature_fns=[
        off_work,
        is_working_day,
        get_one_hot_key(),
        wrapped_partial(backward_target_median, 6),
        time_window,
        get_route_features()
    ])
    print('task1 extract done')
