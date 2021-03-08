import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Net, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action

        self.shared_fc = nn.Linear(state_dim, 200)
        self.mu = nn.Linear(200, action_dim)
        self.sigma = nn.Linear(200, action_dim)
        self.fc_v = nn.Linear(200, 100)
        self.v = nn.Linear(100, 1)
        
        self.init_net([self.shared_fc, self.mu, self.sigma, self.fc_v, self.v])
        self.dist = torch.distributions.Normal
    
    def init_net(self, layers):
        for layer in layers:
            layer.weight.data.normal_(0, 0.1)
    
    def forward(self, x):
        x = F.relu6(self.shared_fc(x))
        mu = self.max_action * torch.tanh(self.mu(x))
        sigma = F.softplus(self.sigma(x)) + 0.001
        v_value = self.v(F.relu6(self.fc_v(x)))

        return mu, sigma, v_value
