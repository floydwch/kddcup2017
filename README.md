# A Solution for KDD Cup 2017

## Models
The samples are segmented by route with the mean travel time in the time window for the task1, and tollgate/direction with the volume in the time window for the task2. Trained with a blending regression model each task. Besides backward target values(travel time and volume), there are some features related to the sample. See `lib/feature.py` and `task{1,2}/feature.py`.


## Results
### Test1
* Task1: 0.181~0.183 MAPE
* Task2: 0.125~0.131 MAPE


## Install Dependencies
```
pip install -r requirements.txt
```

##  Segment Data
```
python -m task1.segment
python -m task2.segment
```

## Extract Features
```
python -m task1.extract
python -m task2.extract
```

## Train Models
```
python -m task1.train
python -m task2.train
```

## Predictions for the Tasks
Since there are test1 and test2 phases, the default is prediction for test1. To predict for the test2 phase, the data should be re-segmented to integrate the test1 data into the training data. The `KDD_MODE` environment variable should be set to `predict` to do so (and set to `train` to switch back to test1).

The resulting files are placed at `{task1, task2}/submission/`.

### Test1
```
python -m task1.predict
python -m task2.predict
```

### Test2 
```
KDD_MODE=predict python -m task1.segment && python -m task1.extract && python -m task1.train && python -m task1.predict
KDD_MODE=predict python -m task2.segment && python -m task2.extract && python -m task2.train && python -m task2.predict
```

## Customize Models
See `task{1,2}/train.py`.


## Customize Features
See `lib/feature.py` for common features and `task{1,2}/feature.py` for specific features, and enable new features in `task{1,2}/extract.py`.
