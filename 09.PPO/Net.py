import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std):
        super(ActorCritic, self).__init__()
        self.fc1_shared = nn.Linear(state_dim, 64)
        self.fc2_shared = nn.Linear(64, 32)
        self.fc_mu = nn.Linear(32, action_dim)
        self.fc_value = nn.Linear(32, 1)
        self.dist = MultivariateNormal
        self.action_var = torch.full((action_dim,), action_std * action_std)

    def forward(self, x):
        x = F.relu6(self.fc1_shared(x))
        x = F.relu6(self.fc2_shared(x))
        mu = self.fc_mu(x)
        s_value = self.fc_value(x)
        cov_mat = torch.diag(self.action_var)

        return mu, cov_mat, s_value

