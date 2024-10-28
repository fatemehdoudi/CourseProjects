import os
import time
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from NetWork import ResNet
from ImageUtils import parse_record
import torch.optim as optim

import wandb

""" This script defines the training, validation and testing process.
"""

class Cifar(nn.Module):
    def __init__(self, config):
        super(Cifar, self).__init__()
        self.config = config
        self.network = ResNet(
            self.config.resnet_version,
            self.config.resnet_size,
            self.config.num_classes,
            self.config.first_num_filters,
        )
        ### YOUR CODE HERE
        # define cross entropy loss and optimizer
        self.loss_func = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.network.parameters(), lr=5e-2)
        #self.optimizer = optim.SGD(self.network.parameters(), lr=5e-2, momentum=0.9, weight_decay=self.config.weight_decay)
        ### YOUR CODE HERE
    
    def train(self, x_train, y_train, max_epoch):
        self.network.train()
        # Determine how many batches in an epoch
        num_samples = x_train.shape[0]
        num_batches = num_samples // self.config.batch_size

        print('### Training... ###')
        for epoch in range(1, max_epoch+1):
            start_time = time.time()
            # Shuffle
            shuffle_index = np.random.permutation(num_samples)
            curr_x_train = x_train[shuffle_index]
            curr_y_train = y_train[shuffle_index]

            ### YOUR CODE HERE
            # Set the learning rate for this epoch
            # Usage example: divide the initial learning rate by 10 after several epochs
            if epoch % 50 == 0:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] =  param_group['lr']/10
            ### YOUR CODE HERE
            
            for i in range(num_batches):
                ### YOUR CODE HERE
                # Construct the current batch.
                # Don't forget to use "parse_record" to perform data preprocessing.
                # Don't forget L2 weight decay
                x_ = curr_x_train[i*self.config.batch_size : (i+1)*self.config.batch_size]
                y = curr_y_train[i*self.config.batch_size : (i+1)*self.config.batch_size]
                x = torch.stack([torch.tensor(parse_record(x_[i], training=True), dtype=torch.float32) for i in range(len(x_))]).to('cuda')
                y = torch.tensor(y , dtype=torch.long).to('cuda')

                y_pred = self.network(x)
                loss = self.loss_func(y_pred, y)
                ### YOUR CODE HERE
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                #print('Batch {:d}/{:d} Loss {:.6f}'.format(i, num_batches, loss), end='\r', flush=True)
            
            duration = time.time() - start_time
            print('Epoch {:d} Loss {:.6f} Duration {:.3f} seconds.'.format(epoch, loss, duration))
            wandb.log({"loss": loss})

            if epoch % self.config.save_interval == 0:
                self.save(epoch)


    def test_or_validate(self, x, y, checkpoint_num_list):
        self.network.eval()
        print('### Test or Validation ###')
        for checkpoint_num in checkpoint_num_list:
            checkpointfile = os.path.join(self.config.modeldir, 'model-%d.ckpt'%(checkpoint_num))
            self.load(checkpointfile)

            preds = []
            for i in range(x.shape[0]):
                ### YOUR CODE HERE
                in_ = parse_record(x[i], training=False)
                in_ = (torch.from_numpy(in_).unsqueeze(0)).float().to('cuda')
                with torch.no_grad():
                    out_ = self.network(in_)
                #out_ = out_.detach()
                y_pred = torch.argmax(out_)
                preds.append(y_pred.item())
                ### END CODE HERE

            y = torch.tensor(y)
            preds = torch.tensor(preds)
            print('Test accuracy: {:.4f}'.format(torch.sum(preds==y)/y.shape[0]))
    
    def save(self, epoch):
        checkpoint_path = os.path.join(self.config.modeldir, 'model-%d.ckpt'%(epoch))
        os.makedirs(self.config.modeldir, exist_ok=True)
        torch.save(self.network.state_dict(), checkpoint_path)
        print("Checkpoint has been created.")
    
    def load(self, checkpoint_name):
        ckpt = torch.load(checkpoint_name, map_location="cpu")
        self.network.load_state_dict(ckpt, strict=True)
        print("Restored model parameters from {}".format(checkpoint_name))