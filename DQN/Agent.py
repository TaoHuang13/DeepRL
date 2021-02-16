import torch
import torch.nn as nn
import numpy as np
from Config import Config
from QNetwork import DQN
from ReplayBuffer import ReplayBuffer

class DQNAgent():
    def __init__(self, env='CartPole-v0'):
        super(DQNAgent, self).__init__()
        self.config = Config()
        self.learn_pointer = 0
        self.declare_net()
        self.declare_memory()
        self.optimizer = torch.optim.Adam(self.EvalNet.parameters(), lr=self.config.LR)
        self.loss = nn.MSELoss()
        
    def declare_net(self):
        self.EvalNet = DQN(self.config)
        self.TargetNet = DQN(self.config)
        self.TargetNet.load_state_dict(self.EvalNet.state_dict())
        self.EvalNet.to(self.config.DEVICE)
        self.TargetNet.to(self.config.DEVICE)
    
    def declare_memory(self):
        self.buffer = ReplayBuffer(self.config.MEMORY_CAPACITY)
    
    def store_transition(self, s0, a, r, s1):
        self.buffer.push(s0, a, r, s1)

    def sample_transition(self):
        assert len(self.buffer) == self.config.MEMORY_CAPACITY
        return self.buffer.sample(self.config.BATCH_SIZE)

    @torch.no_grad()
    def choose_action(self, s):
        if np.random.random() < self.config.EPSILON:
            return np.random.randint(0, self.config.ACTION_SIZE)
        else:
            S = torch.tensor(s, device=self.config.DEVICE, dtype=torch.float)
            action_prob = self.EvalNet(S)
            return int(torch.argmax(action_prob))
    
    def learn(self):
        # update params of targetnet
        if self.learn_pointer % self.config.UPDATE_INTERVAL == 0:
            self.TargetNet.load_state_dict(self.EvalNet.state_dict())
        self.learn_pointer += 1

        # sample from buffer
        batch_sample = self.sample_transition()
        batch_state, batch_action, batch_reward, batch_next_state = zip(*batch_sample)
        batch_state = torch.tensor(batch_state, device=self.config.DEVICE, dtype=torch.float)
        batch_action = torch.tensor(batch_action, device=self.config.DEVICE, dtype=torch.long).unsqueeze(1)
        batch_reward = torch.tensor(batch_reward, device=self.config.DEVICE, dtype=torch.float).unsqueeze(1)
        batch_next_state = torch.tensor(batch_next_state, device=self.config.DEVICE, dtype=torch.float)

        # Q-eval
        q_eval = self.EvalNet(batch_state).gather(1, batch_action)
        q_next = self.TargetNet(batch_next_state).detach()
        q_target = batch_reward + self.config.GAMMA * q_next.max(1)[0].view(self.config.BATCH_SIZE, 1)

        # backward
        loss = self.loss(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

