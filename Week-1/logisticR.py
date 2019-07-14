#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 11:01:54 2019

@author: handabaldeep
"""
import math
import random
import numpy as np

def sigmoid(x):
    return 1/(1+math.exp(-x))

def sigmoid_der(x):
    return math.exp(x)/(math.exp(x)+1)**2

def lr_gd(input_arr,output_arr,noe,learning_rate):
    weights = []
    for a in range(input_arr.shape[1]+1):
        weights.append(random.uniform(0.1,1.0))
    print(weights)
    
    X_train = input_arr[:input_arr.shape[0]//2,:]
    X_test = input_arr[input_arr.shape[0]//2:,:]
    y_train = output_arr[:input_arr.shape[0]//2]
    y_test = output_arr[input_arr.shape[0]//2:]
    print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
    for e in range(noe):
        for i in range(X_train.shape[0]):
            x = weights[0]
            for a in range(input_arr.shape[1]):
                x += weights[a+1]*X_train[i][a]
            
            fp = sigmoid(x)
            bp = 2*(y_train[i]-fp)*sigmoid_der(x)
            
            for a in range(input_arr.shape[1]):
                weights[a+1] += (learning_rate/X_train.shape[0])*bp*X_train[i][a]
            weights[0] += (learning_rate/X_train.shape[0])*bp
    print(weights)
    
    train_mse = 0
    for i in range(X_train.shape[0]):
        x = weights[0]
        for a in range(input_arr.shape[1]):
            x += weights[a+1]*X_train[i][a]
        fp = sigmoid(x)
        train_mse += (y_train[i]-fp)**2
    train_mse /= X_train.shape[0]
    print(train_mse)
    
    test_mse = 0
    for i in range(X_test.shape[0]):
        x = weights[0]
        for a in range(input_arr.shape[1]):
            x += weights[a+1]*X_test[i][a]
        fp = sigmoid(x)
        test_mse += (y_test[i]-fp)**2
    test_mse /= X_test.shape[0]
    print(test_mse)
    

np.random.seed(12)
num_observations = 5000

x1 = np.random.multivariate_normal([0, 0], [[1, .75],[.75, 1]], num_observations)
x2 = np.random.multivariate_normal([1, 4], [[1, .75],[.75, 1]], num_observations)

simulated_separableish_features = np.vstack((x1, x2)).astype(np.float32)
simulated_labels = np.hstack((np.zeros(num_observations),np.ones(num_observations)))
print(simulated_separableish_features.shape,simulated_labels.shape)
lr_gd(simulated_separableish_features,simulated_labels,500,0.001)