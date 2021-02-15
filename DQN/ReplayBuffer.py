import random

class ReplayBuffer():
    def __init__(self, capacity=2000):
        self.capacity = capacity
        self.memory = []

    def push(self, (s0, a, r, s1)):
        self.memory.append((s0, a, r, s1))
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)