import torch
from torch.functional import Tensor
import torch.nn as nn
import math

""" This script defines the network.
"""
# If you are using PyTorch 2.1 and have extra time to play around, you could try to use the PyTorch JIT feature to compile the model by uncommenting the torch.compile decorator. :)
# Please refer to https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html for more details.
# @torch.compile
class ResNet(nn.Module):
    def __init__(self,
            resnet_version,
            resnet_size,
            num_classes,
            first_num_filters,
        ):
        """
        1. Define hyperparameters.
        Args:
            resnet_version: 1 or 2, If 2, use the bottleneck blocks.
            resnet_size: A positive integer (n).
            num_classes: A positive integer. Define the number of classes.
            first_num_filters: An integer. The number of filters to use for the
                first block layer of the model. This number is then doubled
                for each subsampling block layer.
        
        2. Classify a batch of input images.

        Architecture (first_num_filters = 16):
        layer_name      | start | stack1 | stack2 | stack3 | output      |
        output_map_size | 32x32 | 32X32  | 16x16  | 8x8    | 1x1         |
        #layers         | 1     | 2n/3n  | 2n/3n  | 2n/3n  | 1           |
        #filters        | 16    | 16(*4) | 32(*4) | 64(*4) | num_classes |

        n = #residual_blocks in each stack layer = self.resnet_size
        The standard_block has 2 layers each.
        The bottleneck_block has 3 layers each.
        
        Example of replacing:
        standard_block      conv3-16 + conv3-16
        bottleneck_block    conv1-16 + conv3-16 + conv1-64

        Args:
            inputs: A Tensor representing a batch of input images.
        
        Returns:
            A logits Tensor of shape [<batch_size>, self.num_classes].
        """
        super(ResNet, self).__init__()
        self.resnet_version = resnet_version
        self.resnet_size = resnet_size
        self.num_classes = num_classes
        self.first_num_filters = first_num_filters

        ### YOUR CODE HERE
        # define conv1
        self.start_layer = nn.Conv2d(in_channels = 3, out_channels = self.first_num_filters, kernel_size = 3, stride = 1, padding = 1)
        ### YOUR CODE HERE

        # We do not include batch normalization or activation functions in V2
        # for the initial conv1 because the first block unit will perform these
        # for both the shortcut and non-shortcut paths as part of the first
        # block's projection.
        if self.resnet_version == 1:
            self.batch_norm_relu_start = batch_norm_relu_layer(
                num_features=self.first_num_filters, 
                eps=1e-5, 
                momentum=0.997,
            )
        if self.resnet_version == 1:
            block_fn = standard_block
        else:
            block_fn = bottleneck_block

        self.stack_layers = nn.ModuleList()
        for i in range(3): #n_stack
            filters = self.first_num_filters * (2**i)
            strides = 1 if i == 0 else 2
            self.stack_layers.append(stack_layer(filters, block_fn, strides, self.resnet_size, self.first_num_filters))
        self.output_layer = output_layer(filters*4, self.resnet_version, self.num_classes)

    def forward(self, inputs):
        outputs = self.start_layer(inputs)
        if self.resnet_version == 1:
            outputs = self.batch_norm_relu_start(outputs)
        for i in range(3):
            outputs = self.stack_layers[i](outputs)
        outputs = self.output_layer(outputs)
        return outputs

#############################################################################
# Blocks building the network
#############################################################################

class batch_norm_relu_layer(nn.Module):
    """ Perform batch normalization then relu.
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.997) -> None:
        super(batch_norm_relu_layer, self).__init__()
        ### YOUR CODE HERE
        self.batch_norm_relu = nn.Sequential(
            nn.BatchNorm2d(num_features, eps=eps, momentum=momentum) ,
            nn.ReLU(inplace=True)
        )
        ### YOUR CODE HERE
    def forward(self, inputs: Tensor) -> Tensor:
        ### YOUR CODE HERE
        return self.batch_norm_relu(inputs)
        ### YOUR CODE HERE

class standard_block(nn.Module):
    """ Creates a standard residual block for ResNet.

    Args:
        filters: A positive integer. The number of filters for the first 
            convolution.
        projection_shortcut: The function to use for projection shortcuts
      		(typically a 1x1 convolution when downsampling the input).
		strides: A positive integer. The stride to use for the block. If
			greater than 1, this block will ultimately downsample the input.
        first_num_filters: An integer. The number of filters to use for the
            first block layer of the model.
    """
    def __init__(self, filters, projection_shortcut, strides, first_num_filters) -> None:
        super(standard_block, self).__init__()
        blck_num = first_num_filters
        ### YOUR CODE HERE
        self.projection_shortcut = projection_shortcut if projection_shortcut is not None else None
        if blck_num == 0:
            if filters != 16:
                self.layer1 = nn.Sequential(
                    nn.Conv2d(in_channels=filters // 2, out_channels=filters, kernel_size=3, stride = 2, padding=1, bias=False),
                    nn.BatchNorm2d(num_features=filters),
                    nn.ReLU(inplace=True)
                )
            else:
                self.layer1 = nn.Sequential(
                    nn.Conv2d(in_channels=filters, out_channels=filters, kernel_size=3, stride = 1, padding=1, bias=False),
                    nn.BatchNorm2d(num_features=filters),
                    nn.ReLU(inplace=True)
                )
        elif blck_num >0:
            self.layer1 = nn.Sequential(
                nn.Conv2d(in_channels=filters, out_channels=filters, kernel_size=3, stride = 1, padding=1, bias=False),
                nn.BatchNorm2d(num_features=filters),
                nn.ReLU(inplace=True)
            )
            
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=filters, out_channels=filters, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=filters)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs: Tensor) -> Tensor:
        ### YOUR CODE HERE
        res = self.projection_shortcut(inputs) if (self.projection_shortcut is not None) else inputs

        y1 = self.layer1(inputs)
        y2 = self.layer2(y1) + res
        out = self.relu(y2) 
        return out
        ### YOUR CODE HERE

class bottleneck_block(nn.Module):
    """ Creates a bottleneck block for ResNet.

    Args:
        filters: A positive integer. The number of filters for the first 
            convolution. NOTE: filters_out will be 4xfilters.
        projection_shortcut: The function to use for projection shortcuts
      		(typically a 1x1 convolution when downsampling the input).
		strides: A positive integer. The stride to use for the block. If
			greater than 1, this block will ultimately downsample the input.
        first_num_filters: An integer. The number of filters to use for the
            first block layer of the model.
    """
    def __init__(self, filters, projection_shortcut, strides, first_num_filters) -> None:
        super(bottleneck_block, self).__init__()

        ### YOUR CODE HERE
        # Hint: Different from standard lib implementation, you need pay attention to 
        # how to define in_channel of the first bn and conv of each block based on
        # Args given above.
        blck_num = first_num_filters
        self.projection_shortcut = projection_shortcut
        
        if blck_num==0 and filters == 16:
            self.layer1 = nn.Sequential(
                nn.BatchNorm2d(num_features=filters),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=filters, out_channels=filters, kernel_size=1, stride=strides, bias=False), 
            ) 
        elif blck_num==0 and filters != 16:
            self.layer1 = nn.Sequential(
                nn.BatchNorm2d(num_features=2*filters),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=filters*2, out_channels=filters, kernel_size=1, stride=strides, bias=False),
            ) 
        else:
            self.layer1 = nn.Sequential(
                nn.BatchNorm2d(num_features=filters*4),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=filters*4, out_channels=filters, kernel_size=1, stride=strides, bias=False), 
            )
            
        self.layer2 = nn.Sequential(
            nn.BatchNorm2d(num_features=filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=filters, out_channels=filters, kernel_size=3, stride=1, padding=1, bias=False),
        )
        
        self.layer3 = nn.Sequential(
            nn.BatchNorm2d(num_features=filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=filters, out_channels=4*filters, kernel_size=1, stride=1, bias=False),
        )
        
    def forward(self, inputs: Tensor) -> Tensor:
        ### YOUR CODE HERE
        # The projection shortcut should come after the first batch norm and ReLU
		# since it performs a 1x1 convolution.
        ### YOUR CODE HERE
        if self.projection_shortcut is not None:
            res = self.projection_shortcut(inputs)
        else:
            res = inputs
        y1 = self.layer1(inputs)
        y2 = self.layer2(y1)
        out = self.layer3(y2) + res
        
        return out
        
class stack_layer(nn.Module):
    """ Creates one stack of standard blocks or bottleneck blocks.

    Args:
        filters: A positive integer. The number of filters for the first
			    convolution in a block.
		block_fn: 'standard_block' or 'bottleneck_block'.
		strides: A positive integer. The stride to use for the first block. If
				greater than 1, this layer will ultimately downsample the input.
        resnet_size: #residual_blocks in each stack layer
        first_num_filters: An integer. The number of filters to use for the
            first block layer of the model.
    """
    def __init__(self, filters, block_fn, strides, resnet_size, first_num_filters) -> None:
        super(stack_layer, self).__init__()
        stack_num = int(math.log(filters/first_num_filters, 2))
        
        filters_out = filters * 4 if block_fn is bottleneck_block else filters
        ### END CODE HERE
        # projection_shortcut = ?
        # Only the first block per stack_layer uses projection_shortcut and strides
        projection_shortcut = None
        self.blocks = nn.ModuleList()
        
        if block_fn is standard_block:
            if filters != filters_out or strides != 1:
                projection_shortcut = nn.Sequential(
                    nn.Conv2d(filters//2, filters_out, kernel_size=1, stride=strides, bias=False),
                    nn.BatchNorm2d(filters_out)
                )
            if stack_num>0:
                for i in range(resnet_size):
                    if i == 0:
                        block = block_fn(
                            filters=filters, projection_shortcut=projection_shortcut, strides=strides, first_num_filters=i
                        )
                    else:
                        block = block_fn(
                            filters=filters, projection_shortcut=None, strides=1, first_num_filters=i
                        )
                    self.blocks.append(block)
                    filters = filters_out
            else:
                for i in range(resnet_size):   
                    block = block_fn(
                        filters=filters, projection_shortcut=None, strides=1, first_num_filters=i
                    )
                    self.blocks.append(block)
                    filters = filters_out
        else:
            if stack_num>0:
                for i in range(resnet_size):
                    if i == 0:
                        projection_shortcut = nn.Sequential(
                        nn.Conv2d(
                            filters*2, filters*4, kernel_size=1, stride=strides, bias=False
                        ),
                        nn.BatchNorm2d(filters_out)
                        )
                        
                        block = block_fn(
                            filters=filters, projection_shortcut=projection_shortcut, strides=strides, first_num_filters=i
                        )
                    else:
                        block = block_fn(
                            filters=filters, projection_shortcut=None, strides=1, first_num_filters=i
                        )
                    self.blocks.append(block)
            else:
                for i in range(resnet_size):
                    if i == 0:
                        projection_shortcut = nn.Sequential(
                            nn.Conv2d(
                                16, 16*4, kernel_size=1, stride=strides, bias=False
                            ),
                            nn.BatchNorm2d(filters_out)
                        )
                        block = block_fn(
                            filters=filters, projection_shortcut=projection_shortcut, strides=1, first_num_filters=i
                        )
                    else:
                        block = block_fn(
                            filters=filters, projection_shortcut=None, strides=1, first_num_filters=i
                        )
                    self.blocks.append(block)

    def forward(self, inputs: Tensor) -> Tensor:
        rec_input = inputs
        for block in self.blocks:
            rec_input = block(rec_input)
        return rec_input

class output_layer(nn.Module):
    """ Implement the output layer.

    Args:
        filters: A positive integer. The number of filters.
        resnet_version: 1 or 2, If 2, use the bottleneck blocks.
        num_classes: A positive integer. Define the number of classes.
    """
    def __init__(self, filters, resnet_version, num_classes) -> None:
        super(output_layer, self).__init__()
        # Only apply the BN and ReLU for model that does pre_activation in each
		# bottleneck block, e.g. resnet V2.
        self.pre_activation = 1 if resnet_version == 2 else 0
        if (resnet_version == 2):
            self.bn_relu = batch_norm_relu_layer(filters, eps=1e-5, momentum=0.997)
        if resnet_version == 1:
            filters = filters //4
        ### END CODE HERE
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(filters, num_classes)
        ### END CODE HERE
    
    def forward(self, inputs: Tensor) -> Tensor:
        ### END CODE HERE
        x = self.bn_relu(inputs) if self.pre_activation else inputs
        y1 = self.pool(x)
        y1 = y1.view(y1.size(0), -1)
        return self.fc(y1)
        ### END CODE HERE