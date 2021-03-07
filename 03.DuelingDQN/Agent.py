import torch
import torch.nn as nn
import numpy as np
from Config import Config
from Net import DQN
from ReplayBuffer import ReplayBuffer

class DuelingDQNAgent():
    def __init__(self, env='CartPole-v0'):
        super(DuelingDQNAgent, self).__init__()
        self.config = Config(env=env)
        self.declare_net()
        self.declare_memory()
        self.learn_pointer = 0
        self.optimizer = torch.optim.Adam(self.Net1.parameters(), lr=self.config.LR)
        self.loss = nn.MSELoss()
        
    def declare_net(self):
        self.Net1 = DQN(self.config)
        self.Net2 = DQN(self.config)
        self.Net1.to(self.config.DEVICE)
        self.Net2.to(self.config.DEVICE)
    
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
            action_prob = self.Net1(S)
            return int(torch.argmax(action_prob))
    
    def learn(self):
        if self.learn_pointer % self.config.UPDATE_INTERVAL == 0:
            self.Net2.load_state_dict(self.Net1.state_dict())
        self.learn_pointer += 1

        # sample from buffer
        batch_sample = self.sample_transition()
        batch_state, batch_action, batch_reward, batch_next_state = zip(*batch_sample)
        batch_state = torch.tensor(batch_state, device=self.config.DEVICE, dtype=torch.float)
        batch_action = torch.tensor(batch_action, device=self.config.DEVICE, dtype=torch.long).unsqueeze(1)
        batch_reward = torch.tensor(batch_reward, device=self.config.DEVICE, dtype=torch.float).unsqueeze(1)
        batch_next_state = torch.tensor(batch_next_state, device=self.config.DEVICE, dtype=torch.float)

        # Q-eval
        q_eval = self.Net1(batch_state).gather(1, batch_action)
        q_next1 = self.Net1(batch_next_state).gather(1, batch_action)
        q_next2 = self.Net2(batch_next_state).gather(1, batch_action)
        action_index = q_next1.max(1)[1]

        q_target = torch.zeros((self.config.BATCH_SIZE, 1))
        for i in range(self.config.BATCH_SIZE):
            q_target[i] = batch_reward[i] + self.config.GAMMA * q_next2[i, int(action_index[i].data)]
            
        # backward
        loss = self.loss(q_eval, q_target.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()



