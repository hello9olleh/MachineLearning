from typing import Union, Callable, Tuple, Any, Optional, Dict

import torch
import torchvision.datasets as datasets
from torch import Tensor
from torchvision.transforms import v2
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
import torch.nn.utils.parametrize as parametrize


# Convert the grayscale images to binary images
class Binarize(object):
    def __init__(self, threshold):
        self.threshold = threshold / 255

    def __call__(self, img):
        return (img > self.threshold).to(img.dtype)


# Clip the weights and biases of the model to non-negative values
class Clipper(object):
    def __init__(self, frequency=1):
        self.frequency = frequency

    def __call__(self, module):
        for param in module.parameters():
            param.data.clamp_(0)


class shiftedReLU(nn.ReLU):
    def __init__(self, shift: float = 0.5, inplace: bool = False):
        super(shiftedReLU, self).__init__(inplace)
        self.shift = shift

    def forward(self, input: Tensor) -> Tensor:
        return F.relu(input - self.shift, self.inplace)


class scaledSigmoid(nn.Sigmoid):
    def __init__(self, scale: float = 0.5):
        super(scaledSigmoid, self).__init__()
        self.scale = scale

    def forward(self, input: Tensor) -> Tensor:
        return 1 / (1 + torch.exp((- input + self.scale)/self.scale))


class SoftplusParameterization(nn.Module):
    def forward(self, X):
        return F.softplus(X)


train = datasets.MNIST(root='./data', train=True, download=True)
test = datasets.MNIST(root="./data", train=False, download=True)

# Overwriting train and test dataset just with '0' and '1' classes
train.data = train.data[train.targets <= 1]
test.data = test.data[test.targets <= 1]

train.targets = train.targets[train.targets <= 1]
test.targets = test.targets[test.targets <= 1]

# Down sampling the images from 28x28 to 7x7 and binarize them
transforms = v2.Compose([
    v2.Resize(size=(7, 7), antialias=False),
    v2.ToDtype(torch.int8, scale=True),
    Binarize(128)
])

# Not sure why the 'transforms' is not getting applied above
train.data = transforms(train.data)
test.data = transforms(test.data)

# Converting the dataset to PyTorch DataLoader and converting them to float tensors
train = data.TensorDataset(train.data.unsqueeze(1).float(), train.targets.float())
test = data.TensorDataset(test.data.unsqueeze(1).float(), test.targets.float())

train_loader = data.DataLoader(train, batch_size=64, shuffle=True)
test_loader = data.DataLoader(test, batch_size=64, shuffle=False)


# Defining the MLP architecture
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(49, 32)
        self.fc2 = nn.Linear(32, 1)
        self.srelu = shiftedReLU(shift=0.5)

    def forward(self, x):
        x = x.view(-1, 49)
        # x = F.relu(self.fc1(x))
        x = self.srelu(self.fc1(x))
        x = self.fc2(x)

        return x


# Defining a custom nn module to enforce non-negative weights and biases
class NonNegativeNet(nn.Module):
    def __init__(self):
        super(NonNegativeNet, self).__init__()
        self.w1 = nn.Parameter(torch.randn(32, 49))
        self.b1 = nn.Parameter(torch.randn(32))
        self.w2 = nn.Parameter(torch.randn(1, 32))
        self.b2 = nn.Parameter(torch.randn(1))
        self.srelu = shiftedReLU(shift=0.5)
        self.ssigmoid = scaledSigmoid(scale=6)

    def forward(self, x):
        x = x.view(-1, 49)
        x = torch.matmul(x, torch.exp(self.w1.t())) + torch.exp(self.b1)
        x = self.srelu(x)
        x = torch.matmul(x, torch.exp(self.w2.t())) + torch.exp(self.b2)
        # Sigmoid activation function, then scale the output to [0, 1]
        # x = F.sigmoid(x) * 2 - 1
        x = self.ssigmoid(x)
        return x


# DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
DEVICE = 'cpu'
model = MLP()
# model = NonNegativeNet()
model.to(DEVICE)

criterion = nn.BCEWithLogitsLoss()
# criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
clipper = Clipper(frequency=1)

# Training the model
EPOCHS = 100

for epoch in range(EPOCHS):
    model.train()
    for data, target in train_loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target.unsqueeze(1))
        loss.backward()
        optimizer.step()

    if epoch % clipper.frequency == 0:
        model.apply(clipper)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            # print(output)
            predicted = (output > 0).float()
            total += target.size(0)
            correct += (predicted == target.unsqueeze(1)).sum().item()

    print(f'Epoch {epoch + 1}/{EPOCHS} | Loss: {loss.item()} | Accuracy: {correct / total * 100:.2f}%')

# Updating the weights and biases of the model to exponential values as enforced in the forward pass

# Saving the model
torch.save(model.state_dict(), 'models/fp32_clipped.pth')
