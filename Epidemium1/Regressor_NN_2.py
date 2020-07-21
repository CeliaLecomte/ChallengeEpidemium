	from sklearn.base import BaseEstimator
	from sklearn.pipeline import make_pipeline
	from sklearn.preprocessing import Imputer
	from sklearn.ensemble import *
	from sklearn.svm import *
	from sklearn.neighbors import *
	import numpy as np
	from sklearn.preprocessing import *
	from sklearn.decomposition import PCA
	from keras.models import Sequential
	from keras.optimizers import SGD, RMSprop
	from keras.layers.core import *
	 
	 
	class Regressor(BaseEstimator):
	    def __init__(self):
	        self.model = Sequential([
	            Dense(100, input_dim=68, activation="sigmoid", init="lecun_uniform"),
	            Dense(100, activation="sigmoid"),
	            Dense(100, activation="sigmoid"),
	            Dense(5),
	        ])
	        self.model.compile( optimizer='rmsprop',
	                            loss='mse')
	 

	    def fit(self, X, y):
	        print("Training...")
	        imputer = Imputer()
	        X = imputer.fit_transform(X)
	        normalizer = StandardScaler()
	        X = normalizer.fit_transform(X)
	        print "shape : ", X.shape
	        self.model.fit(X, y, nb_epoch=200, batch_size=100)
	        for layer in self.model.layers:
	            weights = layer.get_weights()
	            print "weights ", weights
	        #return self.reg.fit(X, y)
	 
	    def predict(self, X):
	        imputer = Imputer()
	        X = imputer.fit_transform(X)
	        normalizer = StandardScaler()
	        X = normalizer.fit_transform(X)
	        return self.model.predict(X)
	        #return self.reg.predict(X)

