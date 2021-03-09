import torch
import math
from Net import ActorCritic
from Utils import buffer

class PPOAgent():
    def __init__(self, state_dim, action_dim, gamma, std, eps_clip, kepoch, lr, device):
        self.gamma = gamma
        self.std = std
        self.eps_clip = eps_clip
        self.kepoch = kepoch
        self.device = device

        self.declare_net(state_dim, action_dim)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.loss = torch.nn.MSELoss()
        self.buffer = buffer()
    
    def declare_net(self, state_dim, action_dim):
        self.net = ActorCritic(state_dim, action_dim, self.std)
        self.old_net = ActorCritic(state_dim, action_dim, self.std)
        self.old_net.load_state_dict(self.net.state_dict())

    @torch.no_grad()
    def choose_action(self, state):
        state = torch.FloatTensor(state).view(1, -1).to(self.device)
        mu, cov_mat, _ = self.old_net(state)
        cov_mat = cov_mat.to(self.device)
        dist = self.old_net.dist(mu, cov_mat)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        self.buffer.bs.append(state)
        self.buffer.ba.append(action)
        self.buffer.blogp.append(log_prob)

        return action.cpu().data.detach().flatten()

    def act(self, old_state, old_action):
        mu, cov_mat, state_value = self.net(old_state)
        cov_mat = cov_mat.to(self.device)
        dist = self.net.dist(mu, cov_mat)
        action_prob = dist.log_prob(old_action)
        entropy = dist.entropy()

        return action_prob, state_value, entropy

    def learn(self):
        self.net.train()
        bs, ba, br, blogp, bd = self.buffer.get_atr()
        
        # Monte Carlo estimate of rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(br), reversed(bd)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device).view((-1,1))
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        old_state = torch.stack(bs).to(self.device).detach()
        old_action = torch.stack(ba).to(self.device).detach()
        old_logp = torch.stack(blogp).to(self.device).detach()

        for e in range(self.kepoch):
            new_logp, state_value, entropy = self.act(old_state, old_action)
            state_value = state_value.squeeze(2)
            ratio = torch.exp(new_logp - old_logp.detach())
            advs = rewards - state_value.detach()
            surr1 = ratio * advs
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip)
            loss = -torch.min(surr1, surr2) + 0.5 * self.loss(state_value, state_value) - 0.01 * entropy

            self.opt.zero_grad()
            loss.mean().backward()
            self.opt.step()

        self.old_net.load_state_dict(self.net.state_dict())





        



        



    
        