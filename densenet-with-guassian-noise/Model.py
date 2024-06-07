import os
import time
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from Network import DenseNet
from ImageUtils import parse_record
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import wandb
import torch.nn.functional as F


class Cifar(nn.Module):
    def __init__(self, config):
        super(Cifar, self).__init__()
        self.config = config
        # if self.config.model == 'resnet':
        #     self.network = ResNet(
        #     self.config.resnet_size,
        #     self.config.num_classes,
        #     self.config.first_num_filters)
        # elif self.config.model == 'densenet':
        #     self.network = DenseNet([6,12,24,16], 32, 0.5)
        
        self.network = DenseNet([6,12,24,16], 32, 0.5)
        self.loss_func = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.network.parameters(), lr = 0.1, momentum=0.9, weight_decay=self.config.weight_decay)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=5, factor=0.1)
    
    def train(self, x_train, y_train, x_valid, y_valid, max_epoch):
        self.network.train()
        num_samples = x_train.shape[0]
        num_batches = num_samples // self.config.batch_size

        print('### Training... ###')
        for epoch in range(1, max_epoch+1):
            start_time = time.time()
            shuffle_index = np.random.permutation(num_samples)
            curr_x_train = x_train[shuffle_index]
            curr_y_train = y_train[shuffle_index]
            for i in range(num_batches):
                x_ = curr_x_train[i*self.config.batch_size : (i+1)*self.config.batch_size]
                y = curr_y_train[i*self.config.batch_size : (i+1)*self.config.batch_size]
                x = torch.stack(
                    [torch.tensor(parse_record(x_[i], training=True), dtype=torch.float32) for i in range(len(x_))]).to('cuda')
                y = torch.tensor(y , dtype=torch.long).to('cuda')
                y_pred = self.network(x)
                loss = self.loss_func(y_pred, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            duration = time.time() - start_time
            print('Epoch {:d} Loss {:.6f} Duration {:.3f} seconds.'.format(epoch, loss, duration))
            wandb.log({"loss": loss})
            if epoch % self.config.save_interval == 0: ##change!!!
                self.save(epoch)
                
            with torch.no_grad():
                x_val = torch.stack(
                [torch.tensor(parse_record(x_valid[i], training=False), dtype=torch.float32) for i in range(5*self.config.batch_size)]   
                ).to('cuda')
                
                y_val = torch.tensor(y_valid[:5*self.config.batch_size] , dtype=torch.long).to('cuda')
                
                y_pred_val = self.network(x_val)
                loss_val = self.loss_func(y_pred_val, y_val)
                self.scheduler.step(loss_val)
                            
    def test_or_validate(self, x, y, checkpoint_num_list, isPrivate):
        self.network.eval()
        print('### Test or Validation ###')
        for checkpoint_num in checkpoint_num_list:
            checkpointfile = os.path.join(self.config.modeldir, 'model-%d.ckpt' % (checkpoint_num))
            self.load(checkpointfile)
            preds = []
            prob = []
            for i in range(x.shape[0]):
                in_ = parse_record(x[i], training=False)
                in_ = (torch.from_numpy(in_).unsqueeze(0)).float().to('cuda')
                with torch.no_grad():
                    logits = self.network(in_)
                prob.append(F.softmax(logits, dim=1))
                y_pred = torch.argmax(logits)
                preds.append(y_pred.item())

            if not isPrivate:
                y = torch.tensor(y)
                preds = torch.tensor(preds)
                print('Test accuracy: {:.4f}'.format(torch.sum(preds == y) / y.shape[0]))
        if isPrivate:
            prob_np = torch.stack(prob).squeeze().cpu().numpy()
            preds_np = np.array(preds)
            save_path = os.path.join(self.config.modeldir, 'probabilities.npy')
            np.save(save_path, prob_np)
            save_path1 = os.path.join(self.config.modeldir, 'predictions.npy')
            np.save(save_path1, preds_np)

            
    def save(self, epoch):
        checkpoint_path = os.path.join(self.config.modeldir, 'model-%d.ckpt'%(epoch))
        os.makedirs(self.config.modeldir, exist_ok=True)
        torch.save(self.network.state_dict(), checkpoint_path)
        print("Checkpoint has been created.")
    
    def load(self, checkpoint_name):
        ckpt = torch.load(checkpoint_name, map_location="cpu")
        self.network.load_state_dict(ckpt, strict=True)
        print("Restored model parameters from {}".format(checkpoint_name))