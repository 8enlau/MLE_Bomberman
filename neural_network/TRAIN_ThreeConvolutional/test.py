import torch
import torch.nn as nn
from torch.optim import RMSprop

# Define the model
class MyConvolutionModel(nn.Module):
    def __init__(self):
        super(MyConvolutionModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.fc1 = nn.Linear(32 * 26 * 26, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model and move it to the appropriate device
model = MyConvolutionModel().to(device)

# Set up the optimizer
optimizer = RMSprop(params=model.parameters(), lr=0.001)

# Example output to confirm setup
print(model.fc1.shape)
