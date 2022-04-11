#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 17:07:38 2022

@author: Stephen
"""
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures as pf

class PolyLearner:
    def __init__ (self, degree = 1):
        self.degree = degree
        self.model = LinearRegression()
    def train (self, x, y):
        x = pf(degree = self.degree, include_bias = False).fit_transform(x)
        self.model.fit(x, y)
    def test (self, x):
        x = pf(degree = self.degree, include_bias = False).fit_transform(x)
        return self.model.predict(x)