from torch.nn.functional import conv2d,max_pool2d
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.nn.functional import conv2d, max_pool2d, cross_entropy
from tqdm import tqdm
test_error_rate_4 = []
def convolution_model(X, w_conv1, w_conv2, w_conv3, w_h2, w_o, p_drop_input, p_drop_hidden):
    X = dropout(X, p_drop_input)  # Dropout (step 1) on input

    conv1 = rectify(conv2d(X, w_conv1))  # convolutional layer 1 out
    subsampling_layer1 = max_pool2d(conv1, (2, 2))  # subsampling on convolutional layer 1
    conv_out1 = dropout(subsampling_layer1, p_drop_input)  # Dropout  on first convolutional layer

    conv2 = rectify(conv2d(conv_out1, w_conv2))  # convolutional layer 2 out
    subsampling_layer2 = max_pool2d(conv2, (2, 2))  # subsampling on convolutional layer 2
    conv_out2 = dropout(subsampling_layer2, p_drop_input)  # Dropout  on second convolutional layer

    conv3 = rectify(conv2d(conv_out2, w_conv3))  # convolutional layer 3 out
    subsampling_layer3 = max_pool2d(conv3, (2, 2))  # subsampling on convolutional layer 3
    conv_out3 = dropout(subsampling_layer3, p_drop_input)  # Dropout  on third convolutional layer

    conv_out3_dimension = torch.prod(torch.tensor(conv_out3.shape)[1:])
    conv_out3 = conv_out3.reshape(conv_out3.shape[0], conv_out3_dimension)

    h2 = rectify(conv_out3 @ w_h2)  # Layer 1 out
    h2 = dropout(h2, p_drop_hidden)  # Dropout (step 3) on second hidden layer
    pre_softmax = h2 @ w_o  # Layer 2 out (FINAL)

    return pre_softmax





# Configuration values:
n_epochs = 100
p_drop_input = 0.2
p_drop_hidden = 0.5

# initialize weights
# convolutional layers according to table
w_conv1=init_weights((32,1,5,5))
w_save=w_conv1
w_conv2=init_weights((64,32,5,5))
w_conv3=init_weights((128,64,3,3))
# hidden layer with 128 input and 625 output neurons
w_h2 = init_weights((128, 625))
# hidden layer with 625 neurons
w_o = init_weights((625, 10))
# output shape is (B, 10)


# Update oprimiser to include PRelu params
optimizer = RMSprop(params=[w_conv1, w_conv2,w_conv3,w_h2, w_o])

train_loss_convol = []
test_loss_convol = []
train_error_rate_4 = []
test_error_rate_4 = []
# put this into a training loop over 100 epochs
for epoch in range(n_epochs + 1):
    train_loss_this_epoch = []
    incorrect_train = 0
    total_train = 0
    train_progress = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch}/{n_epochs}", unit=" batch")
    for idx, batch in train_progress:
        x, y = batch
        # feed input through model
        noise_py_x = convolution_model(x,  w_conv1,w_conv2,w_conv3,w_h2, w_o, p_drop_input, p_drop_hidden)

        # reset the gradient
        optimizer.zero_grad()

        # the cross-entropy loss function already contains the softmax
        loss = cross_entropy(noise_py_x, y, reduction="mean")

        train_loss_this_epoch.append(float(loss))

        # compute the gradient
        loss.backward()
        # update weights
        optimizer.step()

        # Error rate calculation
        _, predicted = torch.max(noise_py_x, 1)
        incorrect_train += (predicted != y).sum().item()
        total_train += y.size(0)

    train_loss_convol.append(np.mean(train_loss_this_epoch))
    train_error_rate_4.append(incorrect_train / total_train)

    # test periodically
    if epoch % 10 == 0:
        test_loss_this_epoch = []
        incorrect_test = 0
        total_test = 0

        # no need to compute gradients for validation
        with torch.no_grad():
            for idx, batch in enumerate(test_dataloader):
                x, y = batch
                # dropout rates = 0 so that there is no dropout on test
                noise_py_x = convolution_model(x,  w_conv1,w_conv2,w_conv3,w_h2, w_o, p_drop_input, p_drop_hidden)

                loss = cross_entropy(noise_py_x, y, reduction="mean")
                test_loss_this_epoch.append(float(loss))

                _, predicted = torch.max(noise_py_x, 1)
                incorrect_test += (predicted != y).sum().item()
                total_test += y.size(0)

        test_loss_convol.append(np.mean(test_loss_this_epoch))
        error_rate = incorrect_test / total_test
        test_error_rate_4.append(error_rate)