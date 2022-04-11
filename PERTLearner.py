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

class PERTLearner:

    def __init__(self, leaf_size=1): 
        np.set_printoptions(threshold=np.inf)
        np.set_printoptions(precision=20)
        self.leaf_size = leaf_size
        self.tree = None
     
    
    """
    Public function to train the model.
    
    @param x: The x training features
    @param y: The y training labels
    """
    def train(self, x, y): 
        y = np.array([y]).T # Make y into a colunmn
        x = np.append(x, y, axis=1) # Make y the last column of x 
        self.tree = self.make_subtree(x)
    
        return None
    
    
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
    A helper function which randomly selects the split feature and value at each branch of the tree. 
    
    @param x: The x training data we want to determine the best split feature for.
    @ return the index for the best split feature and the value to split on 
    """
    def calculate_best_split_feature(self, x):
  
        x_features = x[:, :-1]
        y = x[:, -1]
        
        feature = int(np.random.choice(range(x_features.shape[1])))
        
        sample_1 = np.random.choice(x_features[:, feature])
 
        sample_2 = np.random.choice((x_features[:, feature]))
        alpha = np.random.random(1)
        
        split_value = alpha * sample_1 + (1 - alpha)* sample_2
      
        return (feature, split_value)
            

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
         
            current_row = np.zeros((1, 4))
            current_row[0, 2] = 1
            left_split_x = np.zeros((1, 4))
            right_split_x = np.zeros((1, 4))
            
            tries = 0
            while (not np.any(left_split_x) or not np.any(right_split_x)) and tries < 10 : # Retry splitting up to 10 times if get a bad split 
             
                
                split_feature, split_value = self.calculate_best_split_feature(x)
                
                current_row[0, 0] = split_feature
        
                current_row[0, 1] = split_value
                 
                left_split_x = x[x[:, split_feature] <= split_value, :] # Note that do not need to sort x when already have the split value
                
                right_split_x = x[x[:, split_feature] > split_value, :]
                tries += 1
                
            if tries >= 10: # if hit 10 tries, make a leaf instead 
                leaf = np.empty((1, 4))
                leaf[:, :] = np.nan
                y = np.mean(y)
                leaf[0, 1] = y
           
                return leaf
            
            else:
                left_child = self.make_subtree(left_split_x)
                    
                right_child = self.make_subtree(right_split_x)
            
                right_number = left_child.shape[0] + 1
                
                current_row[0, 3] = right_number
                
                concatenated = np.vstack((current_row, left_child, right_child))
     
        return concatenated
    
    





