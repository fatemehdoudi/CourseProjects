'''DenseNet in PyTorch.'''
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseNet(nn.Module):
    def __init__(self, num_blocks, growth_rate, theta):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate
        num_filters = 2 * growth_rate
        in_filters = num_filters
        
        self.start_layer = nn.Conv2d(3, num_filters, kernel_size=3, padding=1, bias=False)
        self.stack_layers = nn.ModuleList()

        for i in range(len(num_blocks)):
            layers = []
            for _ in range(num_blocks[i]):
                layers.append(Bottleneck(in_filters, self.growth_rate))
                in_filters += self.growth_rate
            self.stack_layers.append(nn.Sequential(*layers))
            if i != len(num_blocks) - 1:
                out_filters = int(math.floor(in_filters * theta))
                self.stack_layers.append(Transition(in_filters, out_filters))
                in_filters = out_filters
                        
        self.out_layer = nn.Sequential(
            nn.BatchNorm2d(in_filters),
            nn.ReLU(inplace=True)
        ) 
        self.fc = nn.Linear(in_filters, 10)

    def forward(self, x):
        out = self.start_layer(x)
        for layer in self.stack_layers:
            out = layer(out)
        out = self.out_layer(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
    
class Bottleneck(nn.Module):
    def __init__(self, filters, growth_rate):
        super(Bottleneck, self).__init__()
        self.layer1 = nn.Sequential(
                nn.BatchNorm2d(num_features=filters),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=filters, out_channels=4*growth_rate, kernel_size=1, bias=False), 
            ) 
        self.layer2 = nn.Sequential(
                nn.BatchNorm2d(num_features=4*growth_rate),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=4*growth_rate, out_channels=growth_rate, kernel_size=3, padding=1, bias=False), 
            ) 

    def forward(self, x):
        y = self.layer1(x)
        y = self.layer2(y)
        out = torch.cat([y,x], 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_filters, out_filters):
        super(Transition, self).__init__()
        
        self.layer1 = nn.Sequential(
                nn.BatchNorm2d(num_features=in_filters),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=in_filters, out_channels=out_filters, kernel_size=1, bias=False), 
            ) 
    def forward(self, x):
        out = self.layer1(x)
        out = F.avg_pool2d(out, 2)
        return out



