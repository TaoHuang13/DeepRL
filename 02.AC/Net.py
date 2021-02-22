import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNet(nn.Module):
    def __init__(self, config=None):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(config.STATE_SIZE, 128)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc_action = nn.Linear(128, config.ACTION_SIZE)
        self.fc_action.weight.data.normal_(0, 0.1)
        self.fc_state = nn.Linear(128, 1)
        self.fc_state.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        action_prob = F.softmax(self.fc_action(x))
        state_value = self.fc_state(x)

        return action_prob, state_value