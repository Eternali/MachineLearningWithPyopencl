#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as spio
import os, sys, time, math

from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


# input data file
INPUTFILE = '/home/conrad/Documents/cltests/international-airline-passengers.csv'
# lookback length
LOOKBACK = 2


class BasicSequential:
    
    losses = ['mean_squared_error']
    optimizers = ['adam']

    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def compile(self, loss, optimizer):
        if loss not in self.losses or optimizer not in self.optimizers:
            return False

        return True

    def fit(self, X, Y, epochs, batch_size, verbose):
        pass

    def predict(self, X):
        pass


def load_data(filename):
    dataset = pd.read_csv(filename, usecols=[1], engine='python', skipfooter=3).values.astype('float32')
    # normalize the data with standard deviation (note: this is not going to be in between 0 and 1 (for now))
    mean = np.sum(dataset) / dataset.shape[0]
    dataset = (dataset - mean) ** 2
    deviation = np.sqrt(np.sum(dataset) / dataset.shape[0])
    dataset = (dataset - mean) / deviation
    return dataset


def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(dataset.shape[0] - look_back - 1):
        dataX.append(dataset[i:(i + look_back), 0])
        dataY.append(dataset[i + look_back, 0])

    return np.array(dataX), np.array(dataY)


def main():
    # fix random seed for reproducibility
    np.random.seed(7)
    # load the data from file
    dataset = load_data(INPUTFILE)
    # convert dataset to trainable datapoints
    train_size = int(dataset.shape[0] * 0.67)
    test_size = dataset.shape[0] - train_size
    trainX, trainY = create_dataset(dataset[0:train_size, :], look_back=LOOKBACK)
    testX, testY = create_dataset(dataset[train_size:dataset.shape[0], :], look_back=LOOKBACK)
    
    # reshape input to be [samples, time_steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, LOOKBACK)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

    # make predictions off the trained network
    train_predict = model.predict(trainX)
    test_predict = model.predict(testX)
    # invert predictions(?)
    #train_predict = scalar.inverse_transform(train_predict)
    #trainY = scalar.inverse_transform([trainY])
    #test_predict = scalar.inverse_transform(test_predict)
    #testY = scalar.inverse_transform([testY])
    # calculate root mean squared error
    train_score = math.sqrt(mean_squared_error(trainY[0], train_predict[:,0]))
    test_score = math.sqrt(mean_squared_error(testY[0], test_predict[:,0]))

    print('train score: %.2f RSME' % train_score)
    print('test score: %.2f RSME' % test_score)

    # shift training predictions for plotting
    train_predict_plot = np.empty_like(dataset)
    train_predict_plot[:, :] = np.nan
    train_predict_plot[look_back:len(train_predict)+LOOKBACK, :] = train_predict
    # shift testpredictions for plotting
    test_predict_plot = np.empty_like(dataset)
    test_predict_plot[:, :] = np.nan
    test_predict_plot[len(test_predict)+(LOOKBACK*2)+1:len(dataset)-1, :] = test_predict

    # plot baseline and predictions
    # plt.plot(scalar.inverse_transform(dataset))
    # plt.plot(train_predict_plot)
    # plt.plot(test_predict_plot)
    # plt.show()


if __name__ == '__main__':
    main()

