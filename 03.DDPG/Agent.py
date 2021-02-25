import torch
import torch.nn as nn
import numpy as np
import ReplayBuffer
import Net

class DDPGAgent():
    def __init__(self, state_dim, action_dim, max_action, gamma=0.99, capacity=1000000, tau=0.005, lr=1e-4, batch_size=64, device='cpu', update_iteration=200):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.device = device
        self.update_iteration = update_iteration

        self.buffer = ReplayBuffer.ReplayBuffer(capacity=capacity)
        self.declare_net(state_dim, action_dim, max_action)
        self.declare_optimizer(lr)

        self.num_critic_update = 0
        self.num_actor_update = 0

    def declare_net(self, state_dim, action_dim, max_action):
        self.actor = Net.Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = Net.Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Net.Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Net.Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

    def declare_optimizer(self, lr):
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

    def store_transition(self, s0, a, r, s1, d):
        self.buffer.push(s0, a, r, s1, d)

    def sample_transition(self):
        assert len(self.buffer) >= self.batch_size
        return self.buffer.sample(self.batch_size, self.device)

    def choose_action(self, state):
        state = torch.FloatTensor(state.reshape(1,-1)).to(device=self.device)
        action = self.actor(state).cpu().data.numpy().flatten()

        return action

    def optimize(self, batch_state, batch_action, batch_reward, batch_next_state, batch_done):
        #critic_loss
        q_target = self.critic_target(batch_next_state, self.actor_target(batch_next_state))
        q_target = batch_reward + (batch_done * self.gamma * q_target).detach()
        q_eval = self.critic(batch_state, batch_action)
        critic_loss = nn.MSELoss()(q_eval, q_target)
        #optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        #actor_loss
        actor_loss = -self.critic(batch_state, self.actor(batch_state)).mean()
        #optimize actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def updata_target(self):
        for eval_param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * eval_param + (1 - self.tau) * target_param)

        for eval_param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * eval_param + (1 - self.tau) * target_param)

        self.num_critic_update += 1
        self.num_actor_update += 1

    def learn(self):
        for i in range(self.update_iteration):
            batch_state, batch_action, batch_reward, batch_next_state, batch_done = self.buffer.sample(self.batch_size, self.device)
            self.optimize(batch_state, batch_action, batch_reward, batch_next_state, batch_done)
            self.updata_target()

    def save(self, path):
        torch.save(self.actor.state_dict(), path + 'actor.pth')
        torch.save(self.critic.state_dict(), path + 'critic.pth')

    def load(self, path):
        self.actor.load_state_dict(torch.load(path + 'actor.pth'))
        self.critic.load_state_dict(torch.load(path + 'critic.pth'))
        