
# REGRESSOR

from sklearn.base import BaseEstimator
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import *
from sklearn.preprocessing import *
from sklearn import linear_model
from sklearn import svm


class Regressor(BaseEstimator):
    def __init__(self):
        self.clf1 = [make_pipeline(Imputer(),
                                 GradientBoostingRegressor(n_estimators=5000, max_depth=8)) for _ in range(5)]
        self.clf2 = [make_pipeline(Imputer(strategy='median'),
                                  ExtraTreesRegressor(n_estimators=5000, criterion='mse', max_depth=8,
                                                      min_samples_split=10, min_samples_leaf=1,
                                                      min_weight_fraction_leaf=0.0,
                                                      max_features='auto', max_leaf_nodes=None, bootstrap=False,
                                                      oob_score=False,
                                                      n_jobs=1, random_state=42, verbose=0, warm_start=True)) for _ in range(5)]
        self.clf3 = [make_pipeline(Imputer(),
                                  svm.LinearSVR()) for _ in range(5)]
        self.clf = [linear_model.LinearRegression() for _ in range(5)]
 
        
 
    def fit(self, X_t, y_t):
        self.X_t = X_t
        self.y_t = y_t
        [self.clf1[i].fit(X_t, y_t[:,i]) for i in range(5)]
        [self.clf2[i].fit(X_t, y_t[:,i]) for i in range(5)]
        [self.clf3[i].fit(X_t, y_t[:,i]) for i in range(5)]

 
        y1 = self.clf1.predict(self.X_t) 
        y2 = self.clf2.predict(self.X_t)
        y3 = self.clf3.predict(self.X_t)
        
        d = np.column_stack(y1, y2, y3)
        return self.clf.fit(d, y_t)
 
    def predict(self, X_cv):
        r1 = self.clf1.predict(X_cv)
        r2 = self.clf2.predict(X_cv)
        r3 = self.clf3.predict(X_cv)
        r = np.column_stack(r1, r2, r3)
        return self.clf.predict(r)
