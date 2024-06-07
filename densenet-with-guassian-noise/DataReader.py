import os
import pickle
import numpy as np
from sklearn.utils import shuffle

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def add_noise(image, std_dev=500):
    image = image.reshape((3, 32, 32))  # d x H x W
    image = np.transpose(image, [1, 2, 0])  # H x W x d
    color_mask = np.logical_or(image.mean(axis=2) <= 1.05*np.min(image.mean(axis=2))+5, image.mean(axis=2) >= 0.95 *np.max(image.mean(axis=2)))
    noise = np.random.normal(loc=0, scale=std_dev, size=image.shape)
    noisy_image = image.astype(np.float32)
    noisy_image[color_mask] += noise[color_mask]
    noisy_image = np.clip(noisy_image, 0, 255)
    noisy_image = np.transpose(noisy_image, [2, 0, 1])
    noisy_image = noisy_image.flatten()
    return noisy_image

def load_data(data_dir, noise):
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
            elif 'test_batch' in file:
                data_dict = unpickle(file_dir)
                data = data_dict[b'data']
                labels = data_dict[b'labels']
                x_test = data
                y_test = labels
    
    if noise:
        x_train_noiseless = x_train.copy()
        x_train_noisy = np.array([add_noise(image) for image in x_train])
        x_train = np.vstack((x_train_noiseless, x_train_noisy))
        y_train = np.tile(y_train, 2)
        x_train, y_train = shuffle(x_train, y_train, random_state=42)
        x_test = np.array([add_noise(image) for image in x_test])
    
    return x_train, y_train, x_test, y_test

def train_valid_split(x_train, y_train, train_ratio=0.8):
    split_index = int(x_train.shape[0] * train_ratio)
    x_train_new = x_train[:split_index]
    y_train_new = y_train[:split_index]
    x_valid = x_train[split_index:]
    y_valid = y_train[split_index:]
    return x_train_new, y_train_new, x_valid, y_valid

def load_testing_images(data_dir):
    x_test = np.array([], dtype=np.float32).reshape(0, 3072)
    file_dir = os.path.join(data_dir, 'private_test_images_2024.npy')
    data = np.load(file_dir)
    return data