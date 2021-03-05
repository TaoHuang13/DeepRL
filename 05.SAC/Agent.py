import torch
import torch.nn as nn
import numpy as np
import ReplayBuffer
import Net
from torch.distributions import Normal
from torch.nn.utils import clip_grad_norm_

class SACAgent():
    def __init__(self, state_dim, action_dim, max_action, gamma=0.99, capacity=1000000, tau=0.005, lr=1e-4, batch_size=64, device='cpu', update_iteration=200):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.device = device
        self.update_iteration = update_iteration
        self.max_action = max_action

        self.buffer = ReplayBuffer.ReplayBuffer(capacity=capacity)
        self.declare_net(state_dim, action_dim, max_action)
        self.declare_optimizer(lr)
        self.declare_criterion()

        self.num_update = 0
        self.min_val = torch.tensor(1e-7).float().to(device)

    def declare_net(self, state_dim, action_dim, max_action):
        self.actor = Net.Actor(state_dim, max_action).to(self.device)
        self.critic1 = Net.Critic(state_dim, action_dim).to(self.device)
        self.critic2 = Net.Critic(state_dim, action_dim).to(self.device)

        self.value_net = Net.ValueNet(state_dim).to(self.device)
        self.value_net_target = Net.ValueNet(state_dim).to(self.device)
        self.value_net_target.load_state_dict(self.value_net.state_dict())

    def declare_optimizer(self, lr):
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=lr)
        self.value_net_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=lr)

    def declare_criterion(self):
        self.actor_criterion = nn.MSELoss()
        self.critic1_criterion = nn.MSELoss()
        self.critic2_criterion = nn.MSELoss()
        self.value_net_criterion = nn.MSELoss()
    
    def store_transition(self, s0, a, r, s1, d):
        self.buffer.push(s0, a, r, s1, d)

    def sample_transition(self):
        assert len(self.buffer) >= self.batch_size
        return self.buffer.sample(self.batch_size, self.device)

    def choose_action(self, state):
        state = torch.FloatTensor(state).view((1,-1)).to(self.device)
        mu, log_sigma = self.actor(state)
        sigma = torch.exp(log_sigma)
        dist = Normal(mu, sigma)
        
        z = dist.sample()
        action = torch.tanh(z).detach().cpu().data.numpy()
        return action

    # using the current policy rather than replay buffer
    def sample_action(self, batch_state):
        batch_mu, batch_log_sigma = self.actor(batch_state)
        batch_sigma = torch.exp(batch_log_sigma)
        batch_dist = Normal(batch_mu, batch_sigma)
        noise = Normal(0, 1)

        # see Appendix C of the original paper
        z = noise.sample().to(self.device)
        action = batch_mu + batch_sigma * z 
        bounded_action = torch.tanh(action)
        log_prob = batch_dist.log_prob(action) - torch.sum(torch.log(1 - bounded_action.pow(2) + self.min_val))

        return bounded_action, log_prob

    def optimize_actor(self, min_q_value, log_prob):
        actor_loss = (log_prob - min_q_value).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()

    def optimize_critic(self, estimate_q1, estimate_q2, next_q):
        critic1_loss = self.critic1_criterion(estimate_q1, next_q.detach()).mean() # detach target from graph
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        clip_grad_norm_(self.critic1.parameters(), 0.5) # Avoid grad explosion
        self.critic1_optimizer.step()

        critic2_loss = self.critic2_criterion(estimate_q2, next_q.detach()).mean()
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        clip_grad_norm_(self.critic2.parameters(), 0.5)
        self.critic2_optimizer.step()

    def optimize_value_net(self, state_value, min_q_value, log_prob):
        value_loss = self.value_net_criterion(state_value, (min_q_value - log_prob).detach()).mean()
        self.value_net_optimizer.zero_grad()
        value_loss.backward()
        clip_grad_norm_(self.value_net.parameters(), 0.5)
        self.value_net_optimizer.step()

    def update_target(self):
        for eval_param, target_param in zip(self.value_net.parameters(), self.value_net_target.parameters()):
            target_param.data.copy_(self.tau * eval_param + (1 - self.tau) * target_param)

    def learn(self):
        for i in range(self.update_iteration):
            batch_state, batch_action, batch_reward, batch_next_state, batch_done = self.buffer.sample(self.batch_size, self.device)
            bounded_action, log_prob = self.sample_action(batch_state)

            estimate_v = self.value_net(batch_state)
            target_v = self.value_net_target(batch_next_state)
            estimate_q1 = self.critic1(batch_state, batch_action)
            estimate_q2 = self.critic2(batch_state, batch_action)
            target_q = batch_reward + (batch_done * self.gamma * target_v)
            min_q = torch.min(self.critic1(batch_state, bounded_action), self.critic2(batch_state, bounded_action))

            self.optimize_actor(min_q, log_prob)
            self.optimize_critic(estimate_q1, estimate_q2, target_q)
            self.optimize_value_net(estimate_v, min_q, log_prob)
            
            self.update_target()
            self.num_update += 1

    def save(self, path):
        torch.save(self.actor.state_dict(), path + 'actor.pth')
        torch.save(self.critic1.state_dict(), path + 'critic1.pth')
        torch.save(self.critic2.state_dict(), path + 'critic2.pth')
        torch.save(self.value_net.state_dict(), path + 'value_net.pth')

    def load(self, path):
        self.actor.load_state_dict(torch.load(path + 'actor.pth'))
        self.critic1.load_state_dict(torch.load(path + 'critic1.pth'))
        self.critic2.load_state_dict(torch.load(path + 'critic2.pth'))
        self.value_net.load_state_dict(torch.load(path + 'value_net.pth'))
            





    