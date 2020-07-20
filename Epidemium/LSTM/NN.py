from keras.models import Sequential
from keras.layers import Dense, CuDNNLSTM, Dropout, Activation, TimeDistributed
from keras.callbacks import TensorBoard
import numpy as np
from DataPreparation.Utils import *

class LstmNN():
    def __init__(self, windowSize, nbFeatures, neurons = (50, 256), dropout=0.5):
        self.windowSize = windowSize
        self.neurons = neurons
        self.nbFeatures = nbFeatures
        self.model = Sequential()
        self.model.add(CuDNNLSTM(neurons[0], input_shape = (windowSize, nbFeatures), return_sequences=True))
        self.model.add(Dropout(0.2))
        #self.model.add(CuDNNLSTM(neurons[1], return_sequences=True))
        self.model.add(TimeDistributed(Dense(1))) 
        self.model.add(Activation("linear"))
        self.model.compile(loss="mae", optimizer="adam", metrics=['mse', 'mape'])
        self.model.summary()
    
    def rollingWindowPrediction(self, startingWindow, nPred):
        rollingWindow = startingWindow
        pred = np.zeros((nPred, self.nbFeatures))
        for i in range(nPred):
            p = self.model.predict(threeDimInput(rollingWindow))
            pred[i] = p
            rollingWindow = np.concatenate((rollingWindow, p), axis=0)[1:]
        return to2D(np.array(pred))