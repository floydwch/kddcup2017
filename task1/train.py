# -*- coding: utf-8 -*-
import os

from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline

from lib.training import routine
from lib.model import Model


if __name__ == '__main__':
    estimators = [
        ExtraTreesRegressor(
            n_estimators=15,
            criterion='mae',
            min_samples_split=.01,
            bootstrap=True,
            n_jobs=-1,
            random_state=0
        ),
        Pipeline([
            ('scaler', RobustScaler()),
            ('svr', SVR(C=20, gamma=.3, epsilon=.1))
        ])
    ]

    model = Model(estimators, [.45, .55])

    routine(
        dir_path=os.path.dirname(__file__),
        task_id=1,
        model=model,
        mode=os.environ.get('KDD_MODE', 'train')
    )
    print('task1 train done')
