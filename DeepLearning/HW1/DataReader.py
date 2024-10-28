import numpy as np

"""This script implements the functions for reading data.
"""

def load_data(filename):
    """Load a given txt file.

    Args:
        filename: A string.

    Returns:
        raw_data: An array of shape [n_samples, 256].
        labels : An array of shape [n_samples,].
        
    """
    data= np.load(filename)
    x= data['x']
    y= data['y']
    return x, y

def train_valid_split(raw_data, labels, split_index):
	"""Split the original training data into a new training dataset
	and a validation dataset.
	n_samples = n_train_samples + n_valid_samples

	Args:
		raw_data: An array of shape [n_samples, 256].
        labels : An array of shape [n_samples,].
		split_index: An integer.

	"""
	return raw_data[:split_index], raw_data[split_index:], labels[:split_index], labels[split_index:]

def prepare_X(raw_X):
    """Extract features from raw_X as required.

    Args:
        raw_X: An array of shape [n_samples, 256].

    Returns:
        X: An array of shape [n_samples, n_features].
    """
    raw_image = raw_X.reshape((-1, 16, 16))

    # Feature 1: Measure of Symmetry
    L = (raw_image.shape[1])*(raw_image.shape[2])
    flipped = np.flip(raw_image, axis=2)
    Feature_1 = (-((np.abs(raw_image - flipped).sum(axis = 1)).sum(axis = 1))/L)
    # Feature 2: Measure of Intensity
    Feature_2 = ((raw_X.sum(axis = 1))/L)
    # Feature 3: Bias Term. Always 1.
    Feature_3 = (np.ones(raw_X.shape[0]))
    # Stack features together in the following order.
    # [Feature 3, Feature 1, Feature 2]
    X = np.hstack((Feature_3.reshape(-1 , 1), Feature_1.reshape(-1 , 1), Feature_2.reshape(-1 , 1)))
    return X

def prepare_y(raw_y):
    """
    Args:
        raw_y: An array of shape [n_samples,].
        
    Returns:
        y: An array of shape [n_samples,].
        idx:return idx for data label 1 and 2.
    """
    y = raw_y
    idx = np.where((raw_y==1) | (raw_y==2))
    y[np.where(raw_y==0)] = 0
    y[np.where(raw_y==1)] = 1
    y[np.where(raw_y==2)] = 2

    return y, idx