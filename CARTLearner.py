#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 13:59:32 2022
Financial Machine Learning 
Project 3: Decision and Random Trees
2/10/2022
@author: Stephen Crawford

"""

import numpy as np

class CARTLearner:

    def __init__(self, leaf_size=1): 
        np.set_printoptions(threshold=np.inf)
        np.set_printoptions(precision=20)
        self.leaf_size = leaf_size
        
       
     
    
    """
    Public function to train the model.
    
    @param x: The x training features
    @param y: The y training labels
    """
    def train(self, x, y): 
        y = np.array([y]).T # Make y into a colunmn
        x = np.append(x, y, axis=1) # Make y the last column of x 
        self.tree = self.make_subtree(x)
       
       
    
    
    """
    Public function to test the model.
    
    @param x: The test data and accompanying labels stored in a single array with the labels as the last column.
    @return the a 1-D array of predictions for each of the x samples.
    """
    def test(self, x): # Need to move through the rows until we reach a leaf--split off of column etc. 
    
        predictions = []
   
        for row in range(x.shape[0]): # for each x query 
        
            query = x[row,:]
            
            row_index = int(0) 
            
            tree_row = self.tree[row_index,:]
            
            while np.isnan(tree_row[0]) == False: # if not leaf 
            
                tree_row_split_value = tree_row[1]
                tree_row_split_feature = int(tree_row[0])
                query_split_value = query[tree_row_split_feature]
              
                if query_split_value <= tree_row_split_value: # Go to left subtree
                
                    row_index += tree_row[2]
                    tree_row = self.tree[int(row_index), :]
                
                else:
              
                    row_index += tree_row[3]
                    tree_row = self.tree[int(row_index), :]
            
            row_index = 0
            predictions.append(tree_row[1])
    
        predictions = np.array(predictions).flatten()
     
        return predictions   
        
    
    """
    A helper function which calculates the best split feature at each branch of the tree. 
    
    @param x: The x training data we want to determine the best split feature for.
    @ return the index for the best split feature 
    """
    def calculate_best_split_feature(self, x):
    
    
        x_features = x[:, :-1]
        y = x[:, -1]
        split_feature = None
        max_corre = 0
       
        for i in range(0, x_features.shape[1]):
            
            corre = np.corrcoef(x_features[:,i], y)[0,1]
            if abs(corre) >= max_corre:
                
                split_feature = i 
                max_corre = abs(corre)
                
        return split_feature 
     
        
    """
    Helper method to recursively construct a subtree for the parent node. 
     
    @param: X the data we are using to build the subtree--includes Y data as last column 
    @returns: A subtree consisting of all the data in X. 
    """
    def make_subtree(self, x):
      
        y = x[:,-1]

        if len(np.unique(y)) <= 1 or len(np.unique(x[:,:-1])) <= 1 or x.shape[0] <= self.leaf_size: # all y or all x are the same or hit hyperparam stop
           
            leaf = np.empty((1, 4))
            leaf[:, :] = np.nan 
            y = np.mean(y)
            leaf[0, 1] = y
            
            return leaf
            
        else: 
         
            current_row = np.empty((1, 4))
            current_row[:] = np.nan
            current_row[0, 2] = 1
            
            split_feature = self.calculate_best_split_feature(x)
            current_row[0, 0] = split_feature
            
            x_sorted = x[x[:, split_feature].argsort()]
            split_value = np.median(x[:, split_feature])
            
            
            current_row[0, 1] = split_value
            left_split_x = x_sorted[x_sorted[:, split_feature] <= split_value, :] # Was splitting by index but does not work
            right_split_x = x_sorted[x_sorted[:, split_feature] > split_value, :]
        
            if left_split_x.shape[0] == 0 or right_split_x.shape[0] == 0: # Catch edge case of all goiing to left split 
                leaf = np.empty((1, 4))
                leaf[:, :] = np.nan
                values, counts = np.unique(y, return_counts=True)
                y = np.mean(y)
                leaf[0, 1] = y
                return leaf
            
            left_child = self.make_subtree(left_split_x)
            right_child = self.make_subtree(right_split_x)
            right_number = left_child.shape[0] + 1
            current_row[0, 3] = right_number
            concatenated = np.vstack((current_row, left_child, right_child))
     
        return concatenated
    
    





