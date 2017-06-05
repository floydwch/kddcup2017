# -*- coding: utf-8 -*-
import numpy as np


def abs_per_err(y_true, y_pred):
    selected = ~np.isnan(y_true)
    return np.abs((y_true[selected] - y_pred[selected]) / y_true[selected])


def mean_abs_per_err(y_true, y_pred):
    return np.mean(abs_per_err(y_true, y_pred))
