import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import numpy


class DQN(nn.Module):
    '''Current & Target network'''
    def __init__(self, config=None):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(config.STATE_SIZE, 50)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(50,30)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc_adv = nn.Linear(30, config.ACTION_SIZE)
        self.fc_state = nn.Linear(30, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        adv_value = self.fc_adv(x)
        state_value = self.fc_state(x)
        action_value = state_value + adv_value - torch.mean(adv_value)
        
        return adv_value
