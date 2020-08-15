## Define the convolutional neural network architecture

import torch
from torch.autograd import Variable # autograd to calculate the update to the weights in the network
import torch.nn as nn
import torch.nn.functional as F
# The below import is to initialize the weights of the Net
import torch.nn.init as I

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## Defines all the layers of this CNN, the requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## It's suggested that we make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, we have been given a convolutional layer, which we may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        # We apply the NaimishNet consists of 4 convolution2d layers, 4 maxpooling2d layers and 3 dense layers, with sandwiched dropout
        # Reference: https://arxiv.org/pdf/1710.00977.pdf
        self.conv1 = nn.Conv2d(1, 32, 4) #input image size: [1, 224, 224], output size: 32x221x221 
        
        # The size of output of convolutional layer? output width=((W-F+2*P )/S)+1
        # Tensor size or shape: (width, height)
        # Convolution filter size (F): (F_width, F_height)
        # Padding (P): 0
        # Stride (S): 1
        # Note: Zero padding can help keep the size same
        
        ## Note that among the layers to add, consider including: 
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout) to avoid overfitting
        # 
        self.pool1 = nn.MaxPool2d(4, 4)              #32x55x55
       
        self.conv2 = nn.Conv2d(32, 64, 3)            #input: 32x55x55 output: 64x53x53
        self.pool2 = nn.MaxPool2d(2, 2)              #64x26x26

        self.conv3 = nn.Conv2d(64, 128, 2)           #input: 64x26x26 output: 128x25x25
        self.pool3 = nn.MaxPool2d(2, 2)              #128x12x12
        
        self.conv4 = nn.Conv2d(128, 256, 1)          #input: 128x12x12 output: 256x12x12
        self.pool4 = nn.MaxPool2d(2, 2)              #256x6x6

        self.lin1 = nn.Linear(256*6*6,1000)
        self.lin2 = nn.Linear(1000,1000)               
        self.lin3 = nn.Linear(1000,68*2)             #output 136 values
        
    def forward(self, x):
        ## Defines the feedforward behavior of this model
        ## x is the input image and, as an example, here we may choose to include a pool/conv step:
        ## Dropout probability is increased from 0.1 to 0.6 from Dropout1 to Dropout6, with a step size of 0.1.
        drop1 = nn.Dropout(0.1)
        drop2 = nn.Dropout(0.2)
        drop3 = nn.Dropout(0.3)
        drop4 = nn.Dropout(0.4)
        drop5 = nn.Dropout(0.5)
        drop6 = nn.Dropout(0.6)
        
        ## x = self.pool(F.relu(self.conv1(x)))
        x = drop1(self.pool1(F.relu(self.conv1(x))))
        x = drop2(self.pool2(F.relu(self.conv2(x))))
        x = drop3(self.pool3(F.relu(self.conv3(x))))
        x = drop4(self.pool4(F.relu(self.conv4(x))))
        
        # Flatten1 flattens 3d input to 1d output.
        x = x.view(x.size(0), -1) # flatten
        
        x = drop5(F.relu(self.lin1(x)))
        x = drop6(self.lin2(x))
        x = self.lin3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
