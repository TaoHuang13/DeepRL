import torch
import random

class ReplayBuffer():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, s0, a, r, s1, d):
        self.memory.append((s0, a, r, s1, d))
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size, device):
        batch_sample = random.sample(self.memory, batch_size)
        batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*batch_sample)
        
        batch_state = torch.FloatTensor(batch_state).to(device)
        batch_action = torch.FloatTensor(batch_action).unsqueeze(1).to(device)
        batch_reward = torch.FloatTensor(batch_reward).unsqueeze(1).to(device)
        batch_next_state = torch.FloatTensor(batch_next_state).to(device)
        batch_done = torch.FloatTensor(1 - batch_done).to(device)

        return batch_state, batch_action, batch_reward, batch_next_state, batch_done

    def __len__(self):
        return len(self.memory)