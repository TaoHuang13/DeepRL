import os, argparse
import gym
import torch
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter
import Net
from Agent import A3CAgent
from Utils import SharedAdam


parser = argparse.ArgumentParser()
parser.add_argument('--env', default='Pendulum-v0', type=str)
parser.add_argument('--gamma', default=0.9, type=float)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--max_ep', default=3000, type=int)
parser.add_argument('--max_ep_step', default=200, type=int)
parser.add_argument('--update_global', default=5, type=int)
parser.add_argument('--seed', default=False, type=bool)
parser.add_argument('--random_seed', default=9527, type=int)

args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path = './05.A3C/' + args.env + '_Model/'
os.environ["OMP_NUM_THREADS"] = "1"
os.makedirs(path, exist_ok=True)
env = gym.make(args.env).unwrapped
writer = SummaryWriter(path)

if args.seed:
    env.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.shape[0]
MAX_ACTION = env.action_space.high[0]


if __name__ == '__main__':
    glb_net = Net.Net(STATE_DIM, ACTION_DIM, MAX_ACTION)
    glb_net.share_memory()
    optmzr = SharedAdam(glb_net.parameters(), lr=1e-4, betas=(0.95, 0.999))
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

    # parallel
    workers = [A3CAgent(STATE_DIM, ACTION_DIM, MAX_ACTION, device, glb_net, optmzr,\
        global_ep, global_ep_r, res_queue, i, env=args.env, gamma=args.gamma,\
            update_global=args.update_global, max_epi=args.max_ep, max_epi_step=\
                args.max_ep_step) for i in range(mp.cpu_count())]
    [w.start() for w in workers]
    ep = 0
    while True:
        r = res_queue.get()
        if r is not None:
            writer.add_scalar('Reward/Epi_reward', r, global_step=ep)
        else:
            break
        ep += 1
    [w.join() for w in workers]
