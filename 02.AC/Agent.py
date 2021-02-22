import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from Config import Config
from Net import PolicyNet

class ACAgent():
    def __init__(self, env='MountainCar-v0'):
        super(ACAgent, self).__init__()
        self.config = Config(env=env)
        self.declare_net()
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.config.LR)
        self.value_action_state = []
        self.rewards = []

    def declare_net(self):
        self.policy_net = PolicyNet(config=self.config)

    @torch.no_grad()
    def choose_action(self, s):
        S = torch.from_numpy(s).float()
        action_prob, state_value = self.policy_net(S)
        sampler = Categorical(action_prob)
        action = sampler.sample()
        log_prob = sampler.log_prob(action)

        prob_value = self.config.SAVE_ACTION(log_prob, state_value)
        self.value_action_state.append(prob_value)
        return action.item()

    def clear_trajectory(self):
        del self.value_action_state[:]
        del self.rewards[:]

    def learn(self):
        eps = np.finfo(np.float32).eps.item()
        rewards = []
        policy_loss = []
        value_loss = []
        R = 0
        for r in self.rewards[::-1]:
            R = r + self.config.GAMMA * R
            rewards.insert(0, R)
        
        #Normalize the reward
        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + eps)

        for (log_prob, state_value), rtr in zip(self.value_action_state, rewards):
            p_loss = (rtr - state_value.item()) * -log_prob
            policy_loss.append(p_loss)
            v_loss = F.smooth_l1_loss(state_value, torch.tensor([rtr]))
            value_loss.append(v_loss)

        self.optimizer.zero_grad()
        total_loss = torch.tensor(policy_loss).sum() + torch.tensor(value_loss).sum()
        total_loss.requires_grad_(True)
        total_loss.backward()
        self.optimizer.step()
        self.clear_trajectory()




