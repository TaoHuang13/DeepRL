import torch
import torch.nn as nn

from Config import Config
from QNetwork import DQN
from ReplayBuffer import ReplayBuffer

class DQNAgent():
    def __init__(self):
        super(DQNAgent, self).__init__()
        self.config = Config()
        self.MemoryPointer = 0
        self.declare_net()
        self.declare_memory()
        self.optimizer = torch.optim.Adam(self.EvalNet.state_dict(), lr=self.config.LR)
        self.loss = torch.nn.MSELoss()
        
    def declare_net(self):
        self.EvalNet = DQN(self.config)
        self.TargetNet = DQN(self.config)
        self.TargetNet.load_state_dict(self.EvalNet.state_dict())
        self.EvalNet.to(self.config.DEVICE)
        self.TargetNet.to(self.config.DEVICE)
    
    def declare_memory(self, size):
        self.buffer = ReplayBuffer(size=size)
    
    def store_transition(self, (s0, a, r, s1)):
        self.buffer.push((s0, a, r, s1))

    def sample_transition(self):
        pass

    def choose_action(self):
        pass

    def update_param(self):
        pass


