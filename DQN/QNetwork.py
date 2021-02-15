import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import Config
import numpy


class DQN(nn.Module):
    '''Current & Target network'''
    def __init__(self, config=None):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(config.StateSize, 50)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(50,30)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(30, config.ActionSize)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        action_prob = self.out(x)
        
        return action_prob
