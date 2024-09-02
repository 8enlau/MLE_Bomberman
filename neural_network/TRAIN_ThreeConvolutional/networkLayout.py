import torch
from torch.nn.functional import conv2d, max_pool2d, cross_entropy
import numpy as np
import numpy as np
import torch.nn as nn
import json
import matplotlib.pyplot as plt

import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.nn.functional import conv2d, max_pool2d, cross_entropy
from tqdm import tqdm


def rectify(x):
    # Rectified Linear Unit (ReLU)
    return torch.max(torch.zeros_like(x), x)

def dropout(X, p_drop=0.5):
    if 0 < p_drop and p_drop <= 1:
        # Create a binomial mask with the same shape as X
        Phi = torch.bernoulli(p_drop * torch.ones_like(X))
        # Apply dropout
        X_drop = torch.where(Phi == 1, 0, X/(1 - p_drop))
        return X_drop
    # Otherwise return original X
    return X
class NN_model(nn.Module):
    def __init__(self,weights):
        super(NN_model, self).__init__()
        self.w_conv1 = weights[0]
        self.w_conv2 = weights[1]
        self.w_h1 = weights[2]
        self.w_h2 = weights[3]
        self.w_o = weights[-1]
        self.parameters = [self.w_conv1,self.w_conv2,self.w_h1,self.w_h2,self.w_o]

    def parameters(self):
        return([self.w_conv1,self.w_conv2,self.w_h1,self.w_h2,self.w_o])
    def forward(self,X, p_drop_input, p_drop_hidden):
        X = dropout(X, p_drop_input)  # Dropout (step 1) on input

        conv1 = rectify(conv2d(X, self.w_conv1))  # convolutional layer 1 out
       # subsampling_layer1 = max_pool2d(conv1, (1, 1))  # no subsampling on convolutional layer 1
        conv_out1 = dropout(conv1, p_drop_input)  # Dropout  on first convolutional layer

        conv2 = rectify(conv2d(conv_out1, self.w_conv2))  # convolutional layer 2 out
        subsampling_layer2 = max_pool2d(conv2, (2, 2))  # subsampling on convolutional layer 2
        conv_out2 = dropout(subsampling_layer2, p_drop_input)  # Dropout  on first convolutional layer

        conv_out2_dimension = torch.prod(torch.tensor(conv_out2.shape)[2:]) #TODO this is different to callbacks layout:
        conv_out2 = conv_out2.reshape(conv_out2.shape[0],conv_out2.shape[1], conv_out2_dimension)

        h1 = rectify(conv_out2 @ self.w_h1)  # Layer 1 out
        h1 = dropout(h1, p_drop_hidden)  # Dropout (step 3) on second hidden layer

        h2 = rectify(h1 @ self.w_h2)  # Layer 2 out
        h2 = dropout(h2, p_drop_hidden)  # Dropout (step 3) on second hidden layer

        pre_softmax = h2 @ self.w_o  # Layer 2 out (FINAL)
        return pre_softmax.reshape(pre_softmax.shape[0],6)


def testing():
    X=torch.tensor([[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
          [-1, -4.5, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, -3.5, -1],
          [-1, 0, -1, 0, -1, 1, -1, 1, -1, 1, -1, 0, -1, 1, -1, 0, -1],
          [-1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1],
          [-1, 1, -1, 0, -1, 0, -1, 1, -1, 0, -1, 1, -1, 1, -1, 0, -1],
          [-1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, -1],
          [-1, 0, -1, 1, -1, 0, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
          [-1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, -1],
          [-1, 1, -1, 1, -1, 1, -1, 1, -1, 0, -1, 0, -1, 0, -1, 1, -1],
          [-1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, -1],
          [-1, 0, -1, 1, -1, 0, -1, 0, -1, 1, -1, 1, -1, 1, -1, 1, -1],
          [-1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, -1],
          [-1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 0, -1],
          [-1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, -1],
          [-1, 0, -1, 0, -1, 1, -1, 1, -1, 0, -1, 0, -1, 1, -1, 0, -1],
          [-1, -6.5, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, -2.5, -1],
          [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]])
    X=X.reshape ( -1 ,17,17)
    def init_weights(shape):
        # Kaiming He initialization (a good initialization is important)
        # https://arxiv.org/abs/1502.01852
        std = np.sqrt(2. / shape[0])
        w = torch.randn(size=shape) * std
        w.requires_grad = True
        return w
    w_conv1 = init_weights((32, 1, 3, 3))
    w_conv2 = init_weights((6,32,3,3))
    w_h1 = init_weights((36, 625))
    # hidden layer with 128 input and 625 output neurons
    w_h2 = init_weights((625, 128))
    # hidden layer with 625 neurons
    w_o = init_weights((128, 1))
    result = convolution_model(X, [w_conv1,w_conv2,w_h1,w_h2,w_o], 0, 0)
    print(result)
    result = result.reshape(1,6)
    print(result)

    # TODO does dropout make sense in this setting?

