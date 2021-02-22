import torch 
import gym
from collections import namedtuple

class Config():
    '''Configuration'''
    def __init__(self, env='MountainCar-v0'):
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ENV = gym.make(env)
        self.STATE_SIZE, self.ACTION_SIZE = self.get_size()
        self.GAMMA = 0.995
        self.LR = 0.01
        self.SAVE_ACTION = namedtuple('SavedActions',['probs', 'action_values'])

    def get_size(self):
        StateSize, ActionSize = self.ENV.observation_space.shape[0], self.ENV.action_space.n
        return StateSize, ActionSize