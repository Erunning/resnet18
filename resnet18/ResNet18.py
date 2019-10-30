import sys
import numpy as np
import torch
import os
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.datasets import CIFAR10
from utils1 import train, conv3x3

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

class residual_block(nn.Module):
    def __init__(self, in_channel, out_channel, same_shape=True):
        super().__init__()
        self.same_shape = same_shape
        if self.same_shape :
            stride = 1
        else :
            stride = 2
        self.conv1 = conv3x3(in_channel, out_channel, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = conv3x3(out_channel, out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)
        
        if not self.same_shape:
            self.conv3 = nn.Conv2d(in_channel, out_channel, 1, stride=stride)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.bn1(out), True)
        out = self.conv2(out)
        out = F.relu(self.bn2(out), True)

        if not self.same_shape:
            x = self.conv3(x)
        return F.relu(x + out, True)

class resnet(nn.Module):
    def __init__(self, in_channel, num_classes):
        super().__init__()

        self.block1 = nn.Conv2d(in_channel, 64, 7, 2, 3)

        self.block2 = nn.Sequential(
            nn.MaxPool2d(3, 2, 1),
            residual_block(64, 64),
            residual_block(64, 64)
        )

        self.block3 = nn.Sequential(
            residual_block(64, 128, False),
            residual_block(128, 128)
        )

        self.block4 = nn.Sequential(
            residual_block(128, 256, False),
            residual_block(256, 256)
        )

        self.block5 = nn.Sequential(
            residual_block(256, 512, False),
            residual_block(512, 512),
            nn.AvgPool2d(7)
        )

        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x

def image_transform(x):
    x = x.resize((224, 224), 2)
    x = np.array(x, dtype='float32') / 255
    x = (x - 0.5) / 0.5
    x = x.transpose((2, 0, 1))
    x = torch.from_numpy(x)
    return x

train_set = CIFAR10('./data', train=True, transform=image_transform)
train_data = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
test_set = CIFAR10('./data', train=False, transform=image_transform)
test_data = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)

net = resnet(3, 10)
optimizer = torch.optim.SGD(net.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

train(net, train_data, test_data, 100, optimizer, criterion)