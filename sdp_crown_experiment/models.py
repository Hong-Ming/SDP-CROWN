import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def MNIST_MLP():
	model = nn.Sequential(
		nn.Flatten(),
		nn.Linear(784, 100),
		nn.ReLU(),
		nn.Linear(100, 100),
		nn.ReLU(),
		nn.Linear(100, 10)
	)
	return model

class MNIST_ConvSmall(nn.Module):
    def __init__(self):
        super(MNIST_ConvSmall, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=32*7*7, out_features=100)
        self.fc2 = nn.Linear(in_features=100, out_features=10)
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.flatten(x)
        x_p = torch.relu(self.fc1(x))
        x = self.fc2(x_p)
        return x

class MNIST_ConvLarge(nn.Module):
    def __init__(self):
        super(MNIST_ConvLarge, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=64*7*7, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=512)
        self.fc3 = nn.Linear(in_features=512, out_features=10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
def CIFAR10_CNN_A():
    model = nn.Sequential(
        nn.Conv2d(3, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(32*8*8,100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model

def CIFAR10_CNN_B():
    return nn.Sequential(
        nn.ZeroPad2d((1,2,1,2)),
        nn.Conv2d(3, 32, (5,5), stride=2, padding=0),
        nn.ReLU(),
        nn.Conv2d(32, 128, (4,4), stride=2, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(8192, 250),
        nn.ReLU(),
        nn.Linear(250, 10),
    )

def CIFAR10_CNN_C():
    model = nn.Sequential(
        nn.Conv2d(3, 8, 4, stride=2, padding=0),
        nn.ReLU(),
        nn.Conv2d(8, 16, 4, stride=2, padding=0),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(576, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )
    return model

def CIFAR10_ConvSmall():
    model = nn.Sequential(
        nn.Conv2d(3, 16, 4, stride=2, padding=0),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=0),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(32*6*6,100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model

def CIFAR10_ConvDeep():
    model = nn.Sequential(
        nn.Conv2d(3, 8, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(8*8*8, 100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model

class CIFAR10_ConvLarge(nn.Module):
    def __init__(self):
        super(CIFAR10_ConvLarge, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=64*8*8, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=512)
        self.fc3 = nn.Linear(in_features=512, out_features=10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x