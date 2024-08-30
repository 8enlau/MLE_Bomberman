from torch.nn.functional import conv2d,max_pool2d
import numpy as np
import matplotlib.pyplot as plt
import json
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.nn.functional import conv2d, max_pool2d, cross_entropy
from tqdm import tqdm
from networkLayout import convolution_model

def init_weights(shape):
    # Kaiming He initialization (a good initialization is important)
    # https://arxiv.org/abs/1502.01852
    std = np.sqrt(2. / shape[0])
    w = torch.randn(size=shape) * std
    w.requires_grad = True
    return w
# Configuration values:

# Consistent across all excercises:
train_loss_convol = []
test_loss_convol = []
train_error_rate_4 = []
test_error_rate = []
# initialize weights
# convolutional layers according to table
w_conv1=init_weights((32,1,3,3))
w_save=w_conv1
w_conv2=init_weights((64,32,2,2))
w_conv3=init_weights((128,64,1,1))
# hidden layer with 128 input and 625 output neurons
w_h2 = init_weights((128, 625))
# hidden layer with 625 neurons
w_o = init_weights((625, 6))
# output shape is (B, 10)


# transform images into normalized tensors
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])

with open("testdata","r") as file:
    file_read = json.load(file)
torchData=[]
for i,j in file_read:
    i=torch.tensor(i, dtype=torch.float32)
    i = i.reshape ( -1,17,17)
    j=torch.tensor(j, dtype=torch.float32)
    #j = j.reshape(-1,6)
    torchData.append([i,j])
del file_read
train_dataloader = DataLoader(
    dataset=torchData,
    batch_size=batch_size,
    shuffle=True,
    num_workers=1,
    pin_memory=True,
)

test_dataloader = DataLoader(
    dataset=torchData,
    batch_size=batch_size,
    shuffle=False,
    num_workers=1,
    pin_memory=True,
)


class RMSprop(optim.Optimizer):
    """
    This is a reduced version of the PyTorch internal RMSprop optimizer
    It serves here as an example
    """
    def __init__(self, params, lr=1e-3, alpha=0.5, eps=1e-8):
        defaults = dict(lr=lr, alpha=alpha, eps=eps)
        super(RMSprop, self).__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                grad = p.grad.data
                state = self.state[p]

                # state initialization
                if len(state) == 0:
                    state['square_avg'] = torch.zeros_like(p.data)

                square_avg = state['square_avg']
                alpha = group['alpha']

                # update running averages
                square_avg.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)
                avg = square_avg.sqrt().add_(group['eps'])

                # gradient update
                p.data.addcdiv_(grad, avg, value=-group['lr'])










# Update oprimiser to include PRelu params
optimizer = RMSprop(params=[w_conv1, w_conv2,w_conv3,w_h2, w_o])


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
        _,predicted = torch.max(noise_py_x, 1)
        _, y_max_indices = torch.max(y, 1)
        incorrect_train += (predicted != y_max_indices).sum().item()
        total_train += y.size(0)

    train_loss_convol.append(np.mean(train_loss_this_epoch))
    train_error_rate_4.append(incorrect_train / total_train)

    # test periodically
    if epoch % 10 == 0:
        print("Trainerrorrate: ", train_error_rate_4)
        test_loss_this_epoch = []
        incorrect_test = 0
        total_test = 0

        # no need to compute gradients for validation
        with torch.no_grad():
            for idx, batch in enumerate(test_dataloader):
                x, y = batch
                # dropout rates = 0 so that there is no dropout on test
                noise_py_x = convolution_model(x,  w_conv1,w_conv2,w_conv3,w_h2, w_o, 0, 0)

                loss = cross_entropy(noise_py_x, y, reduction="mean")
                test_loss_this_epoch.append(float(loss))

                _, predicted = torch.max(noise_py_x, 1)
                _, y_max_indices = torch.max(y, 1)

                incorrect_test += (predicted != y_max_indices).sum().item()
                total_test += y.size(0)

        test_loss_convol.append(np.mean(test_loss_this_epoch))
        error_rate = incorrect_test / total_test
        test_error_rate.append(error_rate)
print(train_loss_convol)
print(test_loss_convol)
print(train_error_rate_4)
print(test_error_rate)