import torch 
import gym

class Config():
    '''Configuration'''
    def __init__(self, env='CartPole-v0'):
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ENV = gym.make(env)
        self.STATE_SIZE, self.ACTION_SIZE = self.get_size()
        self.GAMMA = 0.9
        self.EPSILON = 0.1
        self.MEMORY_CAPACITY = 2000
        self.LR = 0.01
        self.BATCH_SIZE = 128
        self.UPDATE_INTERVAL = 100

    def get_size(self):
        StateSize, ActionSize = self.ENV.observation_space.shape[0], self.ENV.action_space.n
        return StateSize, ActionSize
