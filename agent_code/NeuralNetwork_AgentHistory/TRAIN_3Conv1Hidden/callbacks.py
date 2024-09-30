import copy
import numpy as np
import os, time
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
    playField = copy.deepcopy(step["field"])
    for i in step["coins"]:
        playField[i[0]][i[1]] = 10
    selfPlayer = step["self"]
    if selfPlayer[2]:
        playField[selfPlayer[3][0]][selfPlayer[3][1]] = 6
    else:
        playField[selfPlayer[3][0]][selfPlayer[3][1]] = 5
    for i in step["others"]:
        playField[i[3][0]][i[3][1]] = 2 + int(i[2]) * 5 / 10
    for i in step["bombs"]:
        k = i[0][0]
        l = i[0][1]
        if i[1] == 3:
            if playField[k][l] >= 4:
                playField[k][l] = 4
            else:
                playField[k][l] = 0.5
        elif i[1] == 2:
            if playField[k][l] >= 4:
                playField[k][l] *= 0.1
            else:
                playField[k][l] = -((9 - i[1]) - playField[k][l])
        elif i[1] == 1:
            if playField[k][l] > 0:
                playField[k][l] *= -1
            else:
                playField[k][l] = -((9 - i[1]) - playField[k][l])
        else:
            playField[k][l] = -((9 - i[1]) - playField[k][l])

    for index1, i in enumerate(step["explosion_map"]):
        for index2, j in enumerate(i):
            if j == 1:
                playField[index1][index2] = -10
    return ([list(row) for row in zip(*playField)])


def rectify(x):
    # Rectified Linear Unit (ReLU)
    return torch.max(torch.zeros_like(x), x)


def convolution_model(X, w_conv1, w_conv2, w_conv3, w_h1, w_o):
    batch_size = X.shape[0]
    conv1 = rectify(conv2d(X, w_conv1, padding=1))  # convolutional layer 1 out
    conv2 = rectify(conv2d(conv1, w_conv2, padding=1))  # convolutional layer 2 out
    subsampling_layer2 = max_pool2d(conv2, (2, 2))  # subsampling on convolutional layer 2
    conv3 = rectify(conv2d(subsampling_layer2, w_conv3, padding=1))

    subsampling_layer3 = max_pool2d(conv3, (2, 2))  # subsampling on convolutional layer 2
    conv_out3 = subsampling_layer3.reshape(batch_size, -1)
    h1 = rectify(conv_out3 @ w_h1)  # Layer 1 out
    pre_softmax = h1 @ w_o  # Layer 2 out (FINAL)
    return pre_softmax


def setup(self):
    if not os.path.exists("weights.pth"):
        weights = {}
        weights["w_conv1"] = init_weights((32, 1, 3, 3))
        weights["w_conv2"] = init_weights((64, 32, 3, 3))
        weights["w_conv3"] = init_weights((128, 64, 3, 3))

        # hidden layer with 2048 input and 256 output neurons
        weights["w_h1"] = init_weights((2048, 128))
        weights["w_o"] = init_weights((128, 6))
        torch.save(weights, "weights.pth")
    else:
        try:
            weights = torch.load("weights.pth", weights_only=True)
        except FileNotFoundError:
            time.sleep(1)
            weights = torch.load("weights.pth", weights_only=True)
    self.w_conv1 = weights["w_conv1"]
    self.w_conv2 = weights["w_conv2"]
    self.w_conv3 = weights["w_conv3"]
    self.w_h1 = weights["w_h1"]
    self.w_o = weights["w_o"]
    self.model = convolution_model
    self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT']
    self.rewriteGameState = rewrite_round_data


def act(agent, game_state: dict):
    reformedGameState = torch.tensor(agent.rewriteGameState(game_state), dtype=torch.float32)
    reformedGameState = reformedGameState.reshape(1, 17, 17)

    prediction = agent.model(reformedGameState,
                             agent.w_conv1,
                             agent.w_conv2,
                             agent.w_conv3,
                             agent.w_h1,
                             agent.w_o)
    return agent.actions[prediction.argmax()]
