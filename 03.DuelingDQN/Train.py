import gym
import matplotlib.pyplot as plt
import argparse
import copy
import Agent

def reward_func(env, x, x_dot, theta, theta_dot):
    r1 = (env.x_threshold - abs(x))/env.x_threshold - 0.5
    r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
    reward = r1 + r2
    return reward

def main():
    agent = Agent.DuelingDQNAgent(env=ENV)
    env = gym.make(ENV)
    #env = env.unwrapped
    reward_list = []
    #plt.ion()
    fig, ax = plt.subplots()
    
    for i in range(EPISODES):
        state = env.reset()
        ep_reward = 0
        while True:
            #env.render()
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            
            if ENV == 'CartPole-v0':
                x, x_dot, theta, theta_dot = next_state
                reward = reward_func(env, x, x_dot, theta, theta_dot)
            agent.store_transition(state, action, reward, next_state)
            ep_reward += reward
            if agent.buffer.check():
                agent.learn()
            if done:
                print("Episode: {}, reward is {}".format(i, round(ep_reward, 3)))
                break
            state = next_state

        r = copy.copy(reward)
        reward_list.append(r)   

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v0')
    parser.add_argument('--episodes', type=int, default=400)
    args = parser.parse_args()

    ENV = args.env
    EPISODES = args.episodes
    main()