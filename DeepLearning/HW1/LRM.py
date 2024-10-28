#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 12:00:48 2019

@author: 
"""

import numpy as np
import sys
#import wandb

"""This script implements a two-class logistic regression model.
"""

class logistic_regression_multiclass(object):
	
    def __init__(self, learning_rate, max_iter, k):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.k = k 
        
    def fit_miniBGD(self, X, labels, batch_size):
        """Train perceptron model on data (X,y) with mini-Batch GD.

        Args:
            X: An array of shape [n_samples, n_features].
            labels: An array of shape [n_samples,].  Only contains 0,..,k-1.
            batch_size: An integer.

        Returns:
            self: Returns an instance of self.

        Hint: the labels should be converted to one-hot vectors, for example: 1----> [0,1,0]; 2---->[0,0,1].
        """

		### YOUR CODE HERE
        n_samples, n_features = X.shape
        self.W = np.zeros((n_features, self.k))
        y = np.zeros((n_samples , self.k))
        y[np.arange(n_samples), labels.astype(int)] = 1
        for iter in range(self.max_iter):
            idx_set = np.random.choice(n_samples, batch_size, replace=False)
            Grad_sum = np.zeros_like(self.W)
            for i in idx_set:
                Grad_sum += self._gradient(X[i, :], y[i , :])
            Grad_mean = Grad_sum / batch_size
            gradient_norm = np.linalg.norm(Grad_mean)
            #wandb.log({"gradient norm": gradient_norm , "series": "Softmax"})
            if gradient_norm < 0.0005:
                print("Converged after", iter+1, "iterations.")
                break
            self.W -= self.learning_rate * Grad_mean
        return self
		### END YOUR CODE
    

    def _gradient(self, _x, _y):
        """Compute the gradient of cross-entropy with respect to self.W
        for one training sample (_x, _y). This function is used in fit_*.

        Args:
            _x: An array of shape [n_features,].
            _y: One_hot vector. 

        Returns:
            _g: An array of shape [n_features, k]. The gradient of
                cross-entropy with respect to self.W.
        """
		### YOUR CODE HERE
        return np.outer(_x , (self.softmax(np.dot(_x.T, self.W)) - _y.T))
		### END YOUR CODE
    
    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        ### You must implement softmax by youself, otherwise you will not get credits for this part.

		### YOUR CODE HERE
        exp_term = np.exp(x)
        return exp_term / np.sum(exp_term)
		### END YOUR CODE
    
    def get_params(self):
        """Get parameters for this perceptron model.

        Returns:
            W: An array of shape [n_features, k].
        """
        if self.W is None:
            print("Run fit first!")
            sys.exit(-1)
        return self.W


    def predict(self, X):
        """Predict class labels for samples in X.

        Args:
            X: An array of shape [n_samples, n_features].

        Returns:
            preds: An array of shape [n_samples,]. Only contains 0,..,k-1.
        """
		### YOUR CODE HERE
        Prob_mat = self.softmax(np.dot(X , self.W))
        indxs = np.argmax(Prob_mat, axis=1)
        return indxs

		### END YOUR CODE


    def score(self, X, labels):
        """Returns the mean accuracy on the given test data and labels.

        Args:
            X: An array of shape [n_samples, n_features].
            labels: An array of shape [n_samples,]. Only contains 0,..,k-1.

        Returns:
            score: An float. Mean accuracy of self.predict(X) wrt. labels.
        """
		### YOUR CODE HERE
        n_samples, n_features = X.shape
        preds = self.predict(X)
        return np.sum((preds == labels))/n_samples

		### END YOUR CODE

