import gym
import argparse
import os
import torch
from itertools import count
import Agent

def main():
    agent = Agent.ACAgent(env=ENV)
    env = gym.make(ENV)
    env = env.unwrapped
    reward_list = []

    env.seed(1)

    for i in range(EPISODES):
        state = env.reset()

        for t in count():
            #env.render()
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            reward = state[0] + reward
            agent.rewards.append(reward)
            state = next_state
            
            if done:
                print("Episode: {}, number of steps is {}".format(i, t))
                break
            
        agent.learn()
        
        # if i % 100 == 0 and i !=0:
        #     modelPath = './AC/AC_' + args.env + '_Model/Model_training' + str(i) + 'Times.pkl'
        #     torch.save(agent, modelPath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='MountainCar-v0')
    parser.add_argument('--episodes', type=int, default=10000)
    args = parser.parse_args()
    os.makedirs('./AC/AC_' + args.env + '_Model/', exist_ok=True)

    ENV = args.env
    EPISODES = args.episodes
    main()