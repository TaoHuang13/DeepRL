import argparse
import os
import torch
import gym
import numpy as np
from itertools import count
import Agent


parser = argparse.ArgumentParser()
parser.add_argument('--env', default='Pendulum-v0', type=str)
parser.add_argument('--gamma', default='0.99', type=float)
parser.add_argument('--capacity', default='1000000', type=int)
parser.add_argument('--tau', default='0.05', type=float)

parser.add_argument('--noise', default=0.1, type=float)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--update_iteration', default=200, type=int)

parser.add_argument('--episodes', default=100000, type=int)
parser.add_argument('--seed', default=False, type=bool)
parser.add_argument('--random_seed', default=2333, type=int)
parser.add_argument('--log_iteration', default=50, type=int)
parser.add_argument('--load', default=False, type=bool)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs('./DDPG_' + parser.env + '_Model/', exist_ok=True)
env = gym.make(parser.env)

if parser.seed:
    env.seed(parser.random_seed)
    torch.manual_seed(parser.random_seed)
    np.random.seed(parser.random_seed)

STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.shape[0]
MAX_ACTION = env.action_space.high[0]
MIN_ACTION = env.action_space.low[0]


def main():
    agent = Agent.DDPGAgent(STATE_DIM, ACTION_DIM, MAX_ACTION, gamma=parser.gamma, capacity=parser.capacity,\
        tau=parser.tau, lr=parser.lr, batch_size=parser.batch_size, device=device, update_iteration=parser.update_iteration)

    if parser.load: agent.load(path)
    for ep in range(parser.episodes):
        ep_reward = 0
        state = env.reset()
        for t in count():
            action = agent.choose_action(state)
            action = (action + np.random.normal(0, parser.noise, size=ACTION_DIM)).clip(MIN_ACTION, MAX_ACTION)
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward

            agent.buffer.push(state, action, reward, next_state, np.float(done))
            if done:
                print("Episode {}, step is {}, reward is {}".format(ep, t + 1, np.round(ep_reward, 2)))
                break
        
        agent.learn()
        if ep % parser.load_iteration == 0 and ep != 0:
            agent.save(path)

if __name__ == '__main__':
    main()

