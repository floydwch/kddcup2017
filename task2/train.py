# -*- coding: utf-8 -*-
import os

from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LassoCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold

from lib.training import routine
from lib.model import Model


if __name__ == '__main__':
    estimators = [
        ExtraTreesRegressor(
            15,
            criterion='mae',
            max_features=.825,
            min_samples_split=.001,
            min_samples_leaf=.001,
            bootstrap=True,
            n_jobs=-1,
            random_state=0
        ),
        Pipeline([
            ('scaler', RobustScaler()),
            ('knn', KNeighborsRegressor(5, weights='distance', n_jobs=-1))
        ]),
        Pipeline([
            ('scaler', RobustScaler()),
            ('svr', SVR(
                C=160,
                gamma=0.1,
                epsilon=0.1
            ))
        ]),
        Pipeline([
            ('scaler', RobustScaler()),
            ('lasso', LassoCV(
                cv=KFold(4, True, 0),
                n_jobs=-1
            ))
        ])
    ]

    model = Model(estimators, [.45, .05, .45, .05])

    routine(
        dir_path=os.path.dirname(__file__),
        task_id=2,
        model=model,
        mode=os.environ.get('KDD_MODE', 'train')
    )
    print('task2 train done')
