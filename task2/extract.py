# -*- coding: utf-8 -*-
import os

from lib.extraction import routine, wrapped_partial
from lib.feature import (
    backward_target_median,
    off_work, is_working_day, get_one_hot_weekday,
    get_weather_features,
    time_window
)

from .feature import (
    get_one_hot_key, get_tollgate, direction,
    backward_target_sum
)


if __name__ == '__main__':
    routine(os.path.dirname(__file__), [
        off_work,
        is_working_day,
        get_one_hot_key(),
        get_tollgate(),
        direction,
        time_window,
        get_one_hot_weekday(),
        get_weather_features([
            'temperature',
            'rel_humidity',
            'precipitation'
        ]),
        wrapped_partial(backward_target_median, 2),
        backward_target_sum
    ])
    print('task2 extract done')
