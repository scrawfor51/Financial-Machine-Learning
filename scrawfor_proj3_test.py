#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 13:51:22 2022

Financial Machine Learning 
Project 3: Decision and Random Trees
2/10/2022
@author: Stephen Crawwford


Resources: Stackoverflow, Numpy Docs, Python Docs, Professor Byrd. 
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics  import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures as pf
from PERTLearner import PERTLearner as PLearner
from CARTLearner import CARTLearner as CLearner
from PolyLearner import PolyLearner as Poly
from matplotlib import pyplot as plt


plt.xlabel('Leaf Size')
plt.ylabel('RMSE')
plt.xticks(range(1,20))
plt.title('CARTLearner Leaf Size Overfitting')
plt.grid()
rmse_ins = []
rmse_outs = []
hyper = []
np.random.seed(759941)
"""
Functional code to conduct experiment for answering problems related to write up. 

# """

# # Load the Istanbul.csv file and remove the header row and date column.
# data = np.genfromtxt("data/winequality-red.csv", delimiter=",")[1:,1:]
# for i in range(20):
#     ins = []
#     outs = []
#     for j in range(10):
#         # Shuffle the rows and partition some data for testing.
#         x_train, x_test, y_train, y_test = train_test_split(data[:,:-1], data[:,-1], test_size=0.4)
        
#         # Construct our learner.
#         lrn = PLearner(leaf_size=i)
#         lrn.train(x_train, y_train)
        
#         # Test in-sample.
#         y_pred = lrn.test(x_train)
#         rmse_is = mean_squared_error(y_train, y_pred, squared=False)
#         corr_is = np.corrcoef(y_train, y_pred)[0,1]
        
#         # Test out-of-sample.
#         y_pred = lrn.test(x_test)
#         rmse_oos = mean_squared_error(y_test, y_pred, squared=False)
#         corr_oos = np.corrcoef(y_test, y_pred)[0,1]
#         ins.append(rmse_is)
#         outs.append(rmse_oos)
#     rmse_ins.append(np.mean(ins))
#     rmse_outs.append(np.mean(outs))
#     hyper.append(i)
    
# plt.plot(hyper, rmse_ins)
# plt.plot(hyper, rmse_outs)
# plt.legend(['In Sample RMSE', 'Out of Sample RMSE'])
# plt.show()

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from PolyLearner import PolyLearner as Learner

# Load the Istanbul.csv file and remove the header row and date column.
data = np.genfromtxt("data/winequality-red.csv", delimiter=",")[:,:]

# Shuffle the rows and partition some data for testing.
x_train, x_test, y_train, y_test = train_test_split(data[:,:-1], data[:,-1], test_size=0.4)

# Construct our learner.
lrn = PLearner(leaf_size=1)
lrn.train(x_train, y_train)


# Test in-sample.
y_pred = lrn.test(x_train)
rmse_is = mean_squared_error(y_train, y_pred, squared=False)
corr_is = np.corrcoef(y_train, y_pred)[0,1]

# Test out-of-sample.
y_pred = lrn.test(x_test)
rmse_oos = mean_squared_error(y_test, y_pred, squared=False)
corr_oos = np.corrcoef(y_test, y_pred)[0,1]

# Print summary.
print (f"In sample, RMSE: {rmse_is}, Corr: {corr_is}")
print (f"Out of sample, RMSE: {rmse_oos}, Corr: {corr_oos}")
