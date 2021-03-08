import torch
import torch.nn as nn
import numpy as np
import ReplayBuffer
import Net

class TD3Agent():
    def __init__(self, state_dim, action_dim, max_action, gamma=0.99, capacity=1000000, tau=0.005, lr=1e-4, batch_size=64, device='cpu', update_iteration=200, policy_delay=2, noise=0.2, noise_clip=0.5):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.device = device
        self.update_iteration = update_iteration
        self.noise = noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.max_action = max_action

        self.buffer = ReplayBuffer.ReplayBuffer(capacity=capacity)
        self.declare_net(state_dim, action_dim, max_action)
        self.declare_optimizer(lr)

        self.num_critic_update = 0
        self.num_actor_update = 0

    def declare_net(self, state_dim, action_dim, max_action):
        self.actor = Net.Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = Net.Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic1 = Net.Critic(state_dim, action_dim).to(self.device)
        self.critic1_target = Net.Critic(state_dim, action_dim).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())

        self.critic2 = Net.Critic(state_dim, action_dim).to(self.device)
        self.critic2_target = Net.Critic(state_dim, action_dim).to(self.device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())

    def declare_optimizer(self, lr):
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=lr)

    def store_transition(self, s0, a, r, s1, d):
        self.buffer.push(s0, a, r, s1, d)

    def sample_transition(self):
        assert len(self.buffer) >= self.batch_size
        return self.buffer.sample(self.batch_size, self.device)

    def choose_action(self, state):
        state = torch.FloatTensor(state.reshape(1,-1)).to(device=self.device)
        action = self.actor(state).cpu().data.numpy().flatten()

        return action

    def optimize_critic(self, batch_state, batch_action, batch_reward, batch_next_state, batch_done):
        #select next action with noise
        noise = torch.ones_like(batch_action).data.normal_(0, self.noise).to(self.device)
        noise = noise.clamp(-self.noise_clip, self.noise_clip)
        next_action = (self.actor_target(batch_next_state) + noise).clamp(-self.max_action, self.max_action)

        #critic_loss
        q_eval1 = self.critic1(batch_state, batch_action)
        q_eval2 = self.critic2(batch_state, batch_action)
        q_target1 = self.critic1_target(batch_next_state, next_action)
        q_target2 = self.critic2_target(batch_next_state, next_action)
        q_target = torch.min(q_target1, q_target2)
        q_target = batch_reward + (batch_done * self.gamma * q_target).detach()
        critic1_loss = nn.MSELoss()(q_eval1, q_target).mean()
        critic2_loss = nn.MSELoss()(q_eval2, q_target).mean()

        #optimize critic1
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        #optimize critic2
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

    def optimize_actor(self, batch_state):
        #actor_loss
        actor_loss = -self.critic1(batch_state, self.actor(batch_state)).mean()
        #optimize actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def updata_target(self):
        for eval_param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
            target_param.data.copy_(self.tau * eval_param + (1 - self.tau) * target_param)

        for eval_param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
            target_param.data.copy_(self.tau * eval_param + (1 - self.tau) * target_param)

        for eval_param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * eval_param + (1 - self.tau) * target_param)

        self.num_critic_update += 1
        self.num_actor_update += self.policy_delay

    def learn(self):
        for i in range(self.update_iteration):
            batch_state, batch_action, batch_reward, batch_next_state, batch_done = self.buffer.sample(self.batch_size, self.device)
            self.optimize_critic(batch_state, batch_action, batch_reward, batch_next_state, batch_done)
            if i % self.policy_delay == 0:
                self.optimize_actor(batch_state)
                self.updata_target()

    def save(self, path):
        torch.save(self.actor.state_dict(), path + 'actor.pth')
        torch.save(self.critic1.state_dict(), path + 'critic1.pth')
        torch.save(self.critic2.state_dict(), path + 'critic2.pth')

    def load(self, path):
        self.actor.load_state_dict(torch.load(path + 'actor.pth'))
        self.critic1.load_state_dict(torch.load(path + 'critic1.pth'))
        self.critic2.load_state_dict(torch.load(path + 'critic2.pth'))
        