from ImageUtils import parse_record
from DataReader import load_data, train_valid_split, load_testing_images
from Model import Cifar
import wandb
import os
import argparse
import numpy as np
import pickle

    
def configure():
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise", type=int, default=0, 
                        help='Noisy Augmentation?')
    parser.add_argument("--batch_size", type=int, default=512, help='training batch size')
    parser.add_argument("--save_interval", type=int, default=10, 
                        help='save the checkpoint when epoch MOD save_interval == 0')
    parser.add_argument("--weight_decay", type=float, default=1e-4, help='weight decay rate')
    parser.add_argument("--modeldir", type=str, default='model_v1', help='model directory')
    return parser.parse_args()
    

def main(config):
    print("--- Preparing Data ---")
    run_wandb = wandb.init(project="Project_DL")
    
    current_dir = os.getcwd()
    parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
    data_dir = os.path.join(parent_dir, '/data')
    
    data_dir = os.getcwd() +'/data'
    # print('Model: ', config.model)
    print('Noise: ', config.noise)
    
    
    x_train, y_train, x_test, y_test = load_data(data_dir , noise = config.noise)
    x_train_new, y_train_new, x_valid, y_valid = train_valid_split(x_train, y_train)

    model = Cifar(config).cuda()
    
    model.train(x_train_new, y_train_new, x_valid, y_valid, 201)
    
    model.test_or_validate(x_valid, y_valid, [160, 170, 180, 190, 200] , isPrivate = False)
    
    model.train(x_train, y_train, x_valid, y_valid, 10)
    model.test_or_validate(x_test, y_test, [10] , isPrivate = False)
    
    x_pds = load_testing_images(data_dir)
    model.test_or_validate(x_pds, [], [10] , isPrivate = True)
        
if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    config = configure()
    main(config)
    wandb.login()