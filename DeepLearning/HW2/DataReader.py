import os
import pickle
import numpy as np

""" This script implements the functions for reading data.
"""

def load_data(data_dir):
    """ Load the CIFAR-10 dataset.
    Args:
        data_dir: A string. The directory where data batches are stored.
    Returns:
        x_train: An numpy array of shape [50000, 3072].  32x32x3
        (dtype=np.float32)
        y_train: An numpy array of shape [50000,]. 
        (dtype=np.int32)
        x_test: An numpy array of shape [10000, 3072]. 
        (dtype=np.float32)
        y_test: An numpy array of shape [10000,]. 
        (dtype=np.int32)
    """
    ### YOUR CODE HERE
    def unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    x_train = np.array([], dtype=np.float32).reshape(0, 3072)
    y_train = np.array([], dtype=np.int32)

    x_test = np.array([], dtype=np.float32).reshape(0, 3072)
    y_test = np.array([], dtype=np.int32)
    
    files = os.listdir(data_dir)
    for file in files:
        if os.path.isfile(os.path.join(data_dir, file)):
            file_dir = os.path.join(data_dir, file)
            
            if 'data_batch' in file:
                data_dict = unpickle(file_dir)
                data = data_dict[b'data']
                labels = data_dict[b'labels']
                x_train = np.concatenate((x_train , data) , axis = 0)
                y_train = np.concatenate((y_train , labels), axis = 0) 
            elif 'test' in file:
                data_dict = unpickle(file_dir)
                data = data_dict[b'data']
                labels = data_dict[b'labels']
                x_test = data
                y_test = labels
    return x_train, y_train, x_test, y_test

def train_vaild_split(x_train, y_train, split_index=45000):
    """ Split the original training data into a new training dataset
        and a validation dataset.
    Args:
        x_train: An array of shape [50000, 3072].
        y_train: An array of shape [50000,].
        split_index: An integer.

    Returns:
        x_train_new: An array of shape [split_index, 3072].
        y_train_new: An array of shape [split_index,].
        x_valid: An array of shape [50000-split_index, 3072].
        y_valid: An array of shape [50000-split_index,].
    """
    x_train_new = x_train[:split_index]
    y_train_new = y_train[:split_index]
    x_valid = x_train[split_index:]
    y_valid = y_train[split_index:]

    return x_train_new, y_train_new, x_valid, y_valid
