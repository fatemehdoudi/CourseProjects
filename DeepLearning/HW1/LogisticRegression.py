import numpy as np
import sys
import pandas as pd
#import wandb


"""This script implements a two-class logistic regression model.
"""

class logistic_regression(object):
	
    def __init__(self, learning_rate, max_iter):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        

    def fit_BGD(self, X, y):
        """Train perceptron model on data (X,y) with Batch Gradient Descent.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            self: Returns an instance of self.
        """
		### YOUR CODE HERE
        n_samples, n_features = X.shape
        self.W = np.zeros(n_features)
        for iter in range(self.max_iter):
            Grad_sum = np.zeros_like(self.W)
            for i in range(n_samples):
                Grad_sum += self._gradient(X[i , :], y[i])
            Grad_mean = Grad_sum/n_samples
            self.W -= self.learning_rate*Grad_mean
		### END YOUR CODE
        return self

    def fit_miniBGD(self, X, y, batch_size):
        """Train perceptron model on data (X,y) with mini-Batch Gradient Descent.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.
            batch_size: An integer.

        Returns:
            self: Returns an instance of self.
        """
		### YOUR CODE HERE
        n_samples, n_features = X.shape
        self.W = np.zeros(n_features)
        for iter in range(self.max_iter):
            idx_set = np.random.choice(n_samples, batch_size, replace=False)
            Grad_sum = np.zeros_like(self.W)
            for i in idx_set:
                Grad_sum += self._gradient(X[i, :], y[i])
            Grad_mean = Grad_sum / batch_size
            self.W -= self.learning_rate * Grad_mean
            gradient_norm = np.linalg.norm(Grad_mean)
            #wandb.log({"gradient norm": gradient_norm , "series": "Sigmoid"})
            if gradient_norm < 0.0005:
                print("Converged after", iter+1, "iterations.")
                break
        return self

    def fit_SGD(self, X, y):
        """Train perceptron model on data (X,y) with Stochastic Gradient Descent.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            self: Returns an instance of self.
        """
        n_samples, n_features = X.shape
        self.W = np.zeros(n_features)
		### YOUR CODE HERE
        for iter in range(self.max_iter):
            idx = np.random.randint(X.shape[0])
            self.W -= self.learning_rate*self._gradient(X[idx , :], y[idx])
		### END YOUR CODE
        return self

    def _gradient(self, _x, _y):
        """Compute the gradient of cross-entropy with respect to self.W
        for one training sample (_x, _y). This function is used in fit_*.

        Args:
            _x: An array of shape [n_features,].
            _y: An integer. 1 or -1.

        Returns:
            _g: An array of shape [n_features,]. The gradient of
                cross-entropy with respect to self.W.
        """
        return -(_y * _x)/(1 + np.exp(_y*np.dot(self.W.T , _x)))


    def get_params(self):
        """Get parameters for this perceptron model.

        Returns:
            W: An array of shape [n_features,].
        """
        if self.W is None:
            print("Run fit first!")
            sys.exit(-1)
        return self.W

    def predict_proba(self, X):
        """Predict class probabilities for samples in X.

        Args:
            X: An array of shape [n_samples, n_features].

        Returns:
            preds_proba: An array of shape [n_samples, 2].
                Only contains floats between [0,1].
        """
		### YOUR CODE HERE
        return 1 / (1 + np.exp(-np.dot(X, self.W.T)))
		### END YOUR CODE


    def predict(self, X):
        """Predict class labels for samples in X.
    
        Args:
            X: An array of shape [n_samples, n_features].
    
        Returns:
            preds: An array of shape [n_samples,]. Only contains 1 or -1.
        """
        P = self.predict_proba(X)
        return np.where(P >= 0.5, 1, -1)


    def score(self, X, y):
        """Returns the mean accuracy on the given test data and labels.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            score: An float. Mean accuracy of self.predict(X) wrt. y.
        """
		### YOUR CODE HERE
        preds = self.predict(X)
        return np.mean((preds == y))
		### END YOUR CODE
    
    def assign_weights(self, weights):
        self.W = weights
        return self