import os
import pandas as pd
import rampwf as rw
from sklearn.model_selection import ShuffleSplit

problem_title = 'Epidemium cancer mortality rate prediction (2nd RAMP)'
_target_column_names = [
    'g_mColon (C18)',
    'g_mLiver (C22)',
    'g_mGallbladder (C23-24)',
    'g_mColon, rectum and anus (C18-21)',
    'g_mIntestine (C17-21)',
]
# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_regression(
    label_names=_target_column_names)
# An object implementing the workflow
workflow = rw.workflows.FeatureExtractorRegressor()

score_types = [
    rw.score_types.RMSE(name='rmse', precision=1),
]


def _read_data(path, f_name):
    data = pd.read_csv(os.path.join(path, 'data', f_name))
    y_array = data[_target_column_names].values
    X_df = data.drop(_target_column_names, axis=1)
    return X_df, y_array


def get_train_data(path='.'):
    f_name = 'train.csv.bz2'
    return _read_data(path, f_name)


def get_test_data(path='.'):
    f_name = 'test.csv.bz2'
    return _read_data(path, f_name)


def get_cv(X, y):
    cv = ShuffleSplit(n_splits=8, test_size=0.5, random_state=57)
    return cv.split(X, y)
