# -*- coding: utf-8 -*-
"""
Stephen Crawford

Project 4: Ensemble Learning 
Financial Machine Learning 
2/18/2022
"""

import numpy as np



"""
A Bootstrap class for ensemble learning with any number of any compatible learning class.

"""
class BootstrapLearner:
    
    def __init__(self, constituent, kwargs, bags):
        self.kwargs = kwargs
        self.bags = bags
        self.constituent = constituent
        self.learners = self.build_constituents()
       
    """
    Internal function that gets called on construction.
    Creates the requested number of ensemble learners.
    
    @return: A list of the created learners.
    """
    def build_constituents(self):
        learners = []
        for i in range(self.bags):
            learners.append(self.constituent(**self.kwargs))
        return learners
    
    
    """
    A public function which trains all ensemble learners on a random subset of the data. 
    
    @param x: The X features to train on 
    @pararm y: The Y values to train on--must be in the same order and of the same size as the X samples
    """
    def train(self, x, y):
        y = np.array([y]).T # Make y into a colunmn
        x = np.append(x, y, axis=1) # Make y the last column of x 
        for model in self.learners: 
            data_subset = [] 
            for i in range(x.shape[0]):
                data_subset.append(x[np.random.choice(x.shape[0]),:])
            data_subset = np.array(data_subset)
            data_x = data_subset[:, :-1]
            data_y = data_subset[:, -1]
            model.train(data_x, data_y)
    
    
    """
    A public function which tests all ensemble learners on input data and returns an averaged prediction for each test sample.
    
    @param: The X samples to test on. 
    """
    def test(self, x):
     
        results = []
        final_pred = []
        for model in self.learners:

            results.append(model.test(x))
            
        results = np.array(results).T
        
        for row in results[:]:
            final_pred.append(np.mean(row))
       
        return final_pred
            
            