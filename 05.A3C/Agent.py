import gym
import math
import torch
import torch.multiprocessing as mp
import numpy as np
from Net import Net
from Utils import v_wrap, store

class A3CAgent(mp.Process):
    def __init__(self, state_dim, action_dim, max_action, device, glb_net, glb_optmzr,\
        glb_ep, glb_ep_r, res_queue, name, env='Pendulum-v0', gamma=0.99, update_global=5,\
            max_epi=3000, max_epi_step=200):
        super(A3CAgent, self).__init__()
        self.max_action = max_action
        self.gamma = gamma
        self.device = device
        self.update_global = update_global
        self.max_epi = 3000
        self.max_epi_step = max_epi_step

        self.lcl_net = Net(state_dim, action_dim, max_action)
        self.name = 'agent%i' % name
        self.glb_ep, self.glb_ep_r, self.res_queue = glb_ep, glb_ep_r, res_queue
        self.glb_net, self.opt = glb_net, glb_optmzr
        self.env = gym.make(env).unwrapped

    @torch.no_grad()
    def choose_action(self, s):
        s = torch.FloatTensor(s.reshape(1, -1)).to(self.device)
        mu, sigma, _ = self.lcl_net(s)
        dist = self.lcl_net.dist(mu.view(1, ).data, sigma.view(1, ).data)
        action = dist.sample().numpy().clip(-self.max_action, self.max_action)
        
        return action

    def compute_loss(self, batch_s, batch_a, batch_v):
        self.lcl_net.train()
        mu, sigma, target_value = self.lcl_net(batch_s)
        td = batch_v - target_value
        critic_loss = td.pow(2)

        dists = self.lcl_net.dist(mu, sigma)
        log_prob = dists.log_prob(batch_a)
        entropy = 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(dists.scale)    # entropy of Gaussian
        actor_loss = -(log_prob * td.detach() + 0.005 * entropy)
        total_loss = (critic_loss + actor_loss).mean()

        return total_loss

    def push_and_pull(self, done, s_next, bs, ba, br):
        if done:
            v_s_ = 0
        else:
            v_s_ = self.lcl_net(v_wrap(s_next[None, :]))[-1].data.numpy()

        buffer_v_target = []
        for r in br[::-1]:
            v_s_ = r + self.gamma * v_s_
            buffer_v_target.append(v_s_)
        buffer_v_target.reverse()

        total_loss = self.compute_loss(v_wrap(np.vstack(bs)),\
            v_wrap(np.array(ba), dtype=np.int64) if ba[0].dtype == np.int64 else v_wrap(np.vstack(ba)),\
            v_wrap(np.array(buffer_v_target)[:, None]))

        # backpropagation in global net
        self.opt.zero_grad()
        total_loss.backward()
        for lp, gp in zip(self.lcl_net.parameters(), self.glb_net.parameters()):
            gp.grad = lp.grad
        self.opt.step()

        # pull parameters from global net
        self.lcl_net.load_state_dict(self.glb_net.state_dict())

    def record(self, epi_r):
        with self.glb_ep.get_lock():
            self.glb_ep.value += 1
        with self.glb_ep_r.get_lock():
            if self.glb_ep_r.value == 0.:
                self.glb_ep_r.value = epi_r
            else:
                self.glb_ep_r.value = self.glb_ep_r.value * 0.99 + epi_r * 0.01
        
        self.res_queue.put(self.glb_ep_r.value)
        print(
            self.name,
            'Ep:', self.glb_ep.value,
            "| Ep_r: %.0f" % self.glb_ep_r.value,
        )

    def run(self):
        total_step = 0
        while self.glb_ep.value < self.max_epi:
            s = self.env.reset()
            buffer_s, buffer_a, buffer_r = [], [], []
            epi_r = 0
            for t in range(self.max_epi_step):
                action = self.choose_action(s)
                s_next, r, done, _ = self.env.step(action)
                epi_r += r
                store(buffer_s, buffer_a, buffer_r, s, action, r) 

                done = (t == self.max_epi_step - 1)
                if total_step % self.update_global or done:
                    self.push_and_pull(done, s_next, buffer_s, buffer_a, buffer_r)
                    if done:
                        self.record(epi_r)

                s = s_next
                total_step += 1
        
        self.res_queue.put(None)



            

    



