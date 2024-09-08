import importlib
import sys

import numpy as np
import os,time
import torch
from torch.nn.functional import conv2d, max_pool2d, cross_entropy

def init_weights(shape):
    # Kaiming He initialization (a good initialization is important)
    # https://arxiv.org/abs/1502.01852
    std = np.sqrt(2. / shape[0])
    w = torch.randn(size=shape) * std
    w.requires_grad = True
    return w
def rewrite_round_data(step):
    playField = step["field"]
    for i in step["coins"]:
        playField[i[0]][i[1]]= 10
    selfPlayer=step["self"]
    playField[selfPlayer[3][0]][selfPlayer[3][1]]=5+int(selfPlayer[2])*5/10
    for i in step["others"]:
        playField[i[3][0]][i[3][1]]=2+int(i[2])*5/10
    for i in step["bombs"]:
        k=i[0][0]
        l=i[0][1]
        if i[1]==3:
            playField[k][l]=-playField[k][l]
        else:
            if playField[k][l]>1:
                playField[k][l] = -(playField[k][l]+(9-i[1])/10)
            else:
                playField[k][l]=-(19-i[1])
    for index1,i in enumerate(step["explosion_map"]):
        for index2,j in enumerate(i):
            if j==1:
                playField[index1][index2]=-20
    return(playField)
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

def convolution_model(X, w_conv1, w_conv2, w_h1, w_h2, w_o, p_drop_input, p_drop_hidden):
    X = dropout(X, p_drop_input)  # Dropout (step 1) on input
    conv1 = rectify(conv2d(X, w_conv1))  # convolutional layer 1 out
    # subsampling_layer1 = max_pool2d(conv1, (1, 1))  # no subsampling on convolutional layer 1
    conv_out1 = dropout(conv1, p_drop_input)  # Dropout  on first convolutional layer

    conv2 = rectify(conv2d(conv_out1, w_conv2))  # convolutional layer 2 out
    subsampling_layer2 = max_pool2d(conv2, (2, 2))  # subsampling on convolutional layer 2
    conv_out2 = dropout(subsampling_layer2, p_drop_input)  # Dropout  on first convolutional layer

    conv_out2_dimension = torch.prod(torch.tensor(conv_out2.shape)[1:])
    conv_out2 = conv_out2.reshape(conv_out2.shape[0], conv_out2_dimension)

    h1 = rectify(conv_out2 @ w_h1)  # Layer 1 out
    h1 = dropout(h1, p_drop_hidden)  # Dropout (step 3) on second hidden layer

    h2 = rectify(h1 @ w_h2)  # Layer 2 out
    h2 = dropout(h2, p_drop_hidden)  # Dropout (step 3) on second hidden layer

    pre_softmax = h2 @ w_o  # Layer 2 out (FINAL)
    return pre_softmax

def setup(self):
    if not os.path.exists("weights.pth"):
        weights = {}
        weights["w_conv1"] = init_weights((32, 1, 3, 3))
        weights["w_conv2"] = init_weights((6,32,3,3))
        weights["w_h1"] = init_weights((36, 625))
        # hidden layer with 128 input and 625 output neurons
        weights["w_h2"] = init_weights((625, 128))
        # hidden layer with 625 neurons
        weights["w_o"] = init_weights((128, 1))
        # output shape is (B, 10)
        torch.save(weights,"weights.pth")
    else:
        try:
            weights = torch.load("weights.pth",weights_only=True)
        except FileNotFoundError:
            time.sleep(1)
            weights = torch.load("weights.pth",weights_only=True)
    self.w_conv1 = weights["w_conv1"]
    self.w_conv2 = weights["w_conv2"]
    self.w_h1 = weights["w_h1"]
    self.w_h2 = weights["w_h2"]
    self.w_o = weights["w_o"]
    self.model = convolution_model
    self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT']
    self.rewriteGameState = rewrite_round_data

def act(agent, game_state: dict):
    reformedGameState = torch.tensor(agent.rewriteGameState(game_state),dtype=torch.float32)
    reformedGameState = reformedGameState.reshape(1,17,17)

    prediction = agent.model(reformedGameState,
                                agent.w_conv1,
                                agent.w_conv2,
                                agent.w_h1,
                                agent.w_h2,
                                agent.w_o,
                                0,
                                0)
    prediction = prediction.reshape(1,6)
    return agent.actions[prediction.argmax()]
