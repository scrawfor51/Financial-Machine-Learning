#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 13:52:18 2022

@author: Stephen
"""

from BootstrapLearner import BootstrapLearner as BLearner
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics  import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures as pf
from PERTLearner import PERTLearner as PLearner
from CARTLearner import CARTLearner as CLearner
from PolyLearner import PolyLearner as Poly
from matplotlib import pyplot as plt
import time


plt.xlabel('Leaf Size')
plt.ylabel('RMSE')
plt.xticks(range(1,20))
plt.title('CARTLearner Leaf Size Overfitting with 20 Bags')
plt.grid()
rmse_ins = []
rmse_outs = []
hyper = []


"""
Functional code to conduct experiment for answering problems related to write up. 

"""
np.seterr(invalid='ignore')
np.random.seed(759941)
# Load the Istanbul.csv file and remove the header row and date column.
data = np.genfromtxt("data/Istanbul.csv", delimiter=",")[1:,1:]
for i in range(1, 21):
    ins = []
    outs = []
    print("Starting tree building leaf size: ", i)
    start_time = time.time()
    for j in range(10):
        # Shuffle the rows and partition some data for testing.
        x_train, x_test, y_train, y_test = train_test_split(data[:,:-1], data[:,-1], test_size=0.4)
        
        # Construct our learner.
        lrn = BLearner(CLearner, kwargs = {"leaf_size":i}, bags=40)
        lrn.train(x_train, y_train)
        
        # Test in-sample.
        y_pred = lrn.test(x_train)
        rmse_is = mean_squared_error(y_train, y_pred, squared=False)
        corr_is = np.corrcoef(y_train, y_pred)[0,1]
        
        # Test out-of-sample.
        y_pred = lrn.test(x_test)
        rmse_oos = mean_squared_error(y_test, y_pred, squared=False)
        corr_oos = np.corrcoef(y_test, y_pred)[0,1]
        ins.append(rmse_is)
        outs.append(rmse_oos)
    print("Fininished in: ", time.time() - start_time)
    
    rmse_ins.append(np.mean(ins))
    rmse_outs.append(np.mean(outs))
    hyper.append(i)
    
plt.plot(hyper, rmse_ins)
plt.plot(hyper, rmse_outs)
plt.legend(['In Sample RMSE', 'Out of Sample RMSE'])
plt.show()

