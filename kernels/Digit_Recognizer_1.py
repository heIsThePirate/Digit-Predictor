# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 14:46:54 2019

@author: mohit
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import keras
from keras.models import Sequential
from keras.layers import Dense

dataset = pd.read_csv('../dataset/train.csv')
X = dataset.iloc[:, 1:].values
y = dataset.iloc[:, 0].values

y_new = np.zeros((np.shape(y)[0], 10))
k = 0
for i in y:
    y_new[k][i] = 1
    k = k + 1

classifier = Sequential()

classifier.add(Dense(
        units = 25, 
        activation = 'relu',
        kernel_initializer = 'uniform',
        input_dim = 784
        ))

classifier.add(Dense(
        units = 25,
        activation = 'relu',
        kernel_initializer = 'uniform'
        ))

classifier.add(Dense(
        units = 10,
        activation = 'sigmoid',
        kernel_initializer = 'uniform'
        ))

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

classifier.fit(X, y_new, batch_size=10, epochs=100)

test_dataset = pd.read_csv('dataset/test.csv')
X_test = test_dataset.iloc[:, :].values

y_pred = classifier.predict(X_test)
y_pred_actual = np.zeros((np.shape(y_pred)[0], 1))

for i in range(np.shape(y_pred)[0]):
    for j in range(np.shape(y_pred)[1]):
        if y_pred[i][j] == 1:
            y_pred_actual[i] = j

y_pred_actual = pd.DataFrame(y_pred_actual)
y_pred_actual.to_csv('../results/Digit_Recognizer_1.csv', index = None)
#y_test_actual = np.zeros((np.shape(y_test)[0], 1))

#for i in range(np.shape(y_test)[0]):
#    for j in range(np.shape(y_test)[1]):
#        if y_test[i][j] == 1:
#            y_test_actual[i] = j

#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test_actual, y_pred_actual)

#Accuracy = 