import argparse
import os
import torch
import gym
import numpy as np
from itertools import count
from tensorboardX import SummaryWriter
import Agent


parser = argparse.ArgumentParser()
parser.add_argument('--env', default='Pendulum-v0', type=str)
parser.add_argument('--gamma', default=0.99, type=float)
parser.add_argument('--capacity', default=50000, type=int)  # Buffer size
parser.add_argument('--tau', default=0.005, type=float) # target smoothing coefficient

parser.add_argument('--explore_noise', default=0.1, type=float) # exploration noise
parser.add_argument('--policy_noise', default=0.2, type=float)  # policy noise
parser.add_argument('--noise_clip', default=0.5, type=float)
parser.add_argument('--policy_delay', default=2, type=int)  # delayed update for actor
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--batch_size', default=100, type=int)
parser.add_argument('--update_iteration', default=2000, type=int)

parser.add_argument('--episodes', default=1000, type=int)
parser.add_argument('--seed', default=False, type=bool)
parser.add_argument('--random_seed', default=9527, type=int)
parser.add_argument('--log_iteration', default=50, type=int)
parser.add_argument('--load', default=False, type=bool)

args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path = './07.TD3/' + args.env + '_Model/'
os.makedirs(path, exist_ok=True)
env = gym.make(args.env)
writer = SummaryWriter(path)

if args.seed:
    env.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.shape[0]
MAX_ACTION = env.action_space.high[0]
MIN_ACTION = env.action_space.low[0]


def main():
    agent = Agent.TD3Agent(STATE_DIM, ACTION_DIM, MAX_ACTION, gamma=args.gamma, capacity=args.capacity,\
        tau=args.tau, lr=args.lr, batch_size=args.batch_size, device=device, update_iteration=args.update_iteration,\
            policy_delay=args.policy_delay, noise=args.policy_noise, noise_clip=args.noise_clip)

    if args.load: agent.load(path)
    for ep in range(args.episodes):
        ep_reward = 0
        state = env.reset()
        for t in count():
            action = agent.choose_action(state)
            action = (action + np.random.normal(0, args.explore_noise, size=ACTION_DIM)).clip(MIN_ACTION, MAX_ACTION)
            next_state, reward, done, _ = env.step(action)
            
            agent.buffer.push(state, action, reward, next_state, np.float(done))
            if done:
                print("Episode {}, step is {}, reward is {}".format(ep, t + 1, np.round(ep_reward, 2)))
                writer.add_scalar('Reward/Epi_reward', ep_reward, global_step=ep)
                break
            
            state = next_state
            ep_reward += reward
        
        if agent.buffer.check_full():  
            agent.learn()
        if ep % args.log_iteration == 0 and ep != 0:
            agent.save(path)

if __name__ == '__main__':
    main()

