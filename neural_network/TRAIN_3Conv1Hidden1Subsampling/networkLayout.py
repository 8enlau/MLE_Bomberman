import torch
from torch.nn.functional import conv2d, max_pool2d, cross_entropy
import numpy as np
import numpy as np
import torch.nn as nn
from torch.nn.functional import conv2d, max_pool2d
import copy

def rewrite_round_data(step):
    playField = copy.deepcopy(step["field"])
    for i in step["coins"]:
        playField[i[0]][i[1]]= 10
    selfPlayer=step["self"]
    if selfPlayer[2]:
        playField[selfPlayer[3][0]][selfPlayer[3][1]] = 6
    else:
        playField[selfPlayer[3][0]][selfPlayer[3][1]] = 5
    for i in step["others"]:
        playField[i[3][0]][i[3][1]]=2+int(i[2])*5/10
    for i in step["bombs"]:
        k=i[0][0]
        l=i[0][1]
        if i[1]==3:
            if playField[k][l]>=4:
                playField[k][l]=4
            else:
                playField[k][l] = 0.5
        elif i[1]==2:
            if playField[k][l]>=4:
                playField[k][l] *= 0.1
            else:
                playField[k][l]=-((9 - i[1])-playField[k][l])
        elif i[1] == 1:
            if playField[k][l] > 0:
                playField[k][l] *= -1
            else:
                playField[k][l] = -((9 - i[1])-playField[k][l])
        else:
            playField[k][l] = -((9 - i[1]) - playField[k][l])

    for index1,i in enumerate(step["explosion_map"]):
        for index2,j in enumerate(i):
            if j==1:
                playField[index1][index2]=-10
    return([list(row) for row in zip(*playField)])

def rectify(x):
    # Rectified Linear Unit (ReLU)
    return torch.max(torch.zeros_like(x), x)

class NN_model(nn.Module):
    def __init__(self,weights):
        super(NN_model, self).__init__()
        self.w_conv1 = weights[0]
        self.w_conv2 = weights[1]
        self.w_conv3 = weights[2]
        self.w_h1 = weights[3]
        self.w_o = weights[-1]
        self.parameters = [self.w_conv1,self.w_conv2,self.w_conv3,self.w_h1,self.w_o]

    def parameters(self):
        return([self.w_conv1,self.w_conv2,self.w_conv3,self.w_h1,self.w_o])
    def forward(self,X, p_drop_input, p_drop_hidden):
        batch_size = X.shape[0]
        conv1 = rectify(conv2d(X, self.w_conv1, padding=1))  # convolutional layer 1 out
        conv2 = rectify(conv2d(conv1, self.w_conv2))  # convolutional layer 2 out
        conv3 = rectify(conv2d(conv2, self.w_conv3))
        subsampling_layer3 = max_pool2d(conv3, (2, 2))  # subsampling on convolutional layer 2
        conv_out3 = subsampling_layer3.reshape(batch_size, -1)
        h1 = rectify(conv_out3 @ self.w_h1)  # Layer 1 out
        pre_softmax = h1 @ self.w_o  # Layer 2 out (FINAL)
        return pre_softmax

class PrepareData:
    def __init__(self):
        self.rewriteData = rewrite_round_data

    def prepareBasic(self,Data):
        return(self.rewriteData(Data))

    def turn_90Degrees(self,Data,label):
        turnedResults = [label[2],
                         label[3],
                         label[1],
                         label[0],
                         label[4],
                         label[5]]
        turnedField = [list(reversed(col)) for col in zip(*Data)]
        return(turnedField,turnedResults)




if __name__=="__main__":
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

    weights = {}
    weights["w_conv1"] = init_weights((32, 1, 3, 3))
    weights["w_conv2"] = init_weights((64, 32, 3, 3))
    weights["w_conv3"] = init_weights((64, 64, 3, 3))

    # hidden layer with 2048 input and 256 output neurons
    weights["w_h1"] = init_weights((2304, 128))
    weights["w_o"] = init_weights((128, 6))
    w_conv1 = weights["w_conv1"]
    w_conv2 = weights["w_conv2"]
    w_conv3 = weights["w_conv3"]
    w_h1 = weights["w_h1"]
    w_o = weights["w_o"]
    model = NN_model([w_conv1,w_conv2,w_conv3,w_h1,w_o])
    result = model.forward(X,0,0)
    print(result)
