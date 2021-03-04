import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class RewardNet(nn.Module):

    def __init__(self, n_features):

        super(RewardNet, self).__init__()

        self.conv1 = nn.Conv2d(n_features, n_features, 1)
        self.conv2 = nn.Conv2d(n_features, n_features, 1)
        self.conv3 = nn.Conv2d(n_features, 1, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return self.conv3(x)


class FourLayersNet(nn.Module):

    def __init__(self, n_features):

        super(FourLayersNet, self).__init__()

        self.conv1 = nn.Conv2d(n_features, n_features*2, 5, padding = 2)
        self.conv2 = nn.Conv2d(n_features*2, n_features, 3, padding = 1)
        self.conv3 = nn.Conv2d(n_features, n_features, 3, padding = 1)
        self.conv4 = nn.Conv2d(n_features, 1, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return (self.conv4(x))


def adjust_learning_rate(lr, optimizer, step):
    lr = lr * (0.1 ** (step // 50))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr