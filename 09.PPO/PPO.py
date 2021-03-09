import os, argparse
import gym
import torch
import numpy as np
from Agent import PPOAgent
from tensorboardX import SummaryWriter


parser = argparse.ArgumentParser()
parser.add_argument('--env', default='BipedalWalker-v3', type=str)
parser.add_argument('--gamma', default=0.99, type=float)
parser.add_argument('--std', default=0.5, type=float)
parser.add_argument('--lr', default=3e-4, type=float)
parser.add_argument('--max_ep', default=10000, type=int)
parser.add_argument('--max_ep_step', default=1500, type=int)
parser.add_argument('--update_time_steps', default=4000, type=int)
parser.add_argument('--kepoch', default=50, type=int)
parser.add_argument('--eps_clip', default=0.2, type=float)
parser.add_argument('--log_iteration', default=200, type=int)
parser.add_argument('--seed', default=False, type=bool)
parser.add_argument('--random_seed', default=9527, type=int)

args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path = './09.PPO/' + args.env + '_Model/'
os.makedirs(path, exist_ok=True)
env = gym.make(args.env).unwrapped
writer = SummaryWriter(path)

if args.seed:
    env.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.shape[0]

if __name__ == '__main__':
    ppo = PPOAgent(STATE_DIM, ACTION_DIM, args.gamma, args.std, args.eps_clip, args.kepoch, args.lr, device)
    time_step = 0

    for ep in range(args.max_ep):
        running_reward = 0
        state = env.reset()
        for t in range(args.max_ep_step):
            time_step += 1
            action = ppo.choose_action(state)
            next_state, r, done, _ = env.step(action)
            
            ppo.buffer.store_r_d(r, done)
            if time_step % args.update_time_steps == 0:
                ppo.learn()
                ppo.buffer.clear()
                time_step = 0

            running_reward += r
            if done: 
                print("Episode {}, step is {}, reward is {}".format(ep, t + 1, np.round(running_reward, 2)))
                writer.add_scalar('Reward/Epi_reward', running_reward, global_step=ep)
                writer.add_scalar('Steps/Epi_steps', t + 1, global_step=ep)
                break

            state = next_state
        
        # save every 500 episodes
        if ep % args.log_iteration == 0:
            torch.save(ppo.net.state_dict(), path + 'net.pth')

