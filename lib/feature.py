# -*- coding: utf-8 -*-
from itertools import starmap
import datetime

import numpy as np
from sklearn.preprocessing import OneHotEncoder


def get_one_hot_weekday():
    encoder = OneHotEncoder(n_values=[7], sparse=False)
    encoder.fit([[0]])

    def one_hot_weekday(ix, **kargs):
        return encoder.transform([
            [datetime.date(2016, *ix_to_date(ix)).weekday()]
        ])[0].tolist()

    return one_hot_weekday


def get_weather_features(selected_cols):
    def weather_features(ix, weather_df, **kargs):
        weather_query = \
            'date.dt.month == {} & date.dt.day == {} & hour == {}'
        hour = 15 if off_work(ix) else 6
        hour_weather = weather_df.query(
            weather_query.format(*ix_to_date(ix), hour)
        )
        return hour_weather[selected_cols].values[0].tolist() \
            if hour_weather.size \
            else weather_df.query(weather_query.format(
                ix[0], ix[1] - 1, hour
            ))[selected_cols].values[0].tolist()

    return weather_features


def off_work(ix, **kargs):
    return 17 <= ix[2] <= 18


def time_window(ix, **kargs):
    if ix[2] in [8, 17]:
        return ix[3]
    elif ix[2] in [9, 18]:
        return ix[3] + 3


def year_day(ix, **kargs):
    return datetime.date(2016, *ix_to_date(ix)).timetuple().tm_yday


def is_holiday(ix, **kargs):
    date = ix_to_date(ix)
    if date[0] == 9 and 15 <= date[1] <= 17 or \
            date[0] == 10 and 1 <= date[1] <= 7:
        return True
    return False


def is_working_day(ix, **kargs):
    if not is_holiday(ix) and \
        datetime.date(2016, *ix_to_date(ix)).weekday() <= 4 and \
            ix_to_date(ix) not in [(9, 18), (10, 8), (10, 9)]:
        return True
    return False


def backward_targets(n_windows, ix, target_df, **kargs):
    month, day, hour = ix[:3]
    key = ix[-2:]
    return target_df.loc[list(starmap(
        lambda offset, window: (
            month, day, hour - offset, 2 - window, *key
        ),
        (window_to_offset(i) for i in range(n_windows))
    ))].values.tolist()


def backward_target_median(n_windows, ix, target_df, **kargs):
    return np.nan_to_num(
        np.nanmedian(backward_targets(n_windows, ix, target_df))
    )


def window_to_offset(window):
    if window // 3 == 0:
        return 1, window % 3
    return 2, window % 3


def ix_to_date(ix):
    return ix[0], ix[1]
