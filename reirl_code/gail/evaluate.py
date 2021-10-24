import argparse
import gym
import pybulletgym
import gym_simple
import os
import sys
import pickle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from itertools import count
from utils import *


parser = argparse.ArgumentParser(description='Save expert trajectory')
parser.add_argument('--env-name', default="Hopper-v2", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--expert-traj-path', metavar='G',
                    help='name of the expert model')
parser.add_argument('--model-path', metavar='G',
                    help='name of the learned model')
parser.add_argument('--render', action='store_true', default=False,
                    help='render the environment')
parser.add_argument('--seed', type=int, default=1, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--max-expert-state-num', type=int, default=50000, metavar='N',
                    help='maximal number of main iterations (default: 50000)')
parser.add_argument('--noiseE', type=float, default=None, metavar='G')
parser.add_argument('--grid-type', type=int, default=None, metavar='N')
parser.add_argument('--mass-mul', type=float, default=1.0, metavar='G',
                    help="Multiplier for CartPole and Acrobot masses")
parser.add_argument('--len-mul', type=float, default=1.0, metavar='G',
                    help="Multiplier for CartPole and Acrobot lengths")
args = parser.parse_args()

dtype = torch.float64
torch.set_default_dtype(dtype)
if args.env_name == "gridworld-v0":
    env = gym.make("gridworld-v0", prop = args.noiseE, env_type = args.grid_type)
    subfolder = "env"+str(args.grid_type)+"noiseE"+str(args.noiseE)
    if not os.path.isdir(assets_dir(subfolder+"/expert_traj")):
        os.makedirs(assets_dir(subfolder+"/expert_traj"))
if args.env_name == "CartPole-v1" or args.env_name == "Acrobot-v1":
    env = gym.make(args.env_name)
    subfolder = "env"+ args.env_name + "mass" + str(args.mass_mul)+ "len" + str(args.len_mul)
    if not os.path.isdir(assets_dir(subfolder+"/expert_traj")):
        os.makedirs(assets_dir(subfolder+"/expert_traj"))
    if args.env_name == "Acrobot-v1":
        env.env.LINK_LENGTH_1 *= args.len_mul
        env.env.LINK_LENGTH_2 *= args.len_mul
        env.env.LINK_MASS_1 *= args.mass_mul
        env.env.LINK_MASS_2 *= args.mass_mul
    elif args.env_name == "CartPole-v1":
        env.env.masspole *= args.mass_mul
        env.env.masscart *= args.mass_mul
        env.env.length *= args.len_mul
else:
    env = gym.make(args.env_name)
    subfolder = None
env.seed(args.seed)
torch.manual_seed(args.seed)
is_disc_action = len(env.action_space.shape) == 0
state_dim = env.observation_space.shape[0]
_, running_state = pickle.load(open(args.expert_traj_path, "rb"))
policy_net, _, _ = pickle.load(open(args.model_path, "rb"))
running_state.fix = True
expert_traj = []
state_only_expert_traj = []

def main_loop():

    num_steps = 0

    for i_episode in count():

        state = env.reset()
        state = running_state(state)
        print(state.shape)
        reward_episode = 0

        for t in range(10000):
            state_var = tensor(state).unsqueeze(0).to(dtype)
            # choose mean action
            if not is_disc_action:
                action = policy_net(state_var)[0][0].detach().numpy()
            else:
                action = policy_net.select_action(state_var)[0].numpy()
            # action = policy_net.select_action(state_var)[0].cpu().numpy()
            action = int(action) if is_disc_action else action.astype(np.float64)
            next_state, reward, done, _ = env.step(action)
            next_state = running_state(next_state)
            reward_episode += reward
            num_steps += 1

            expert_traj.append(np.hstack([state, action]))
            state_only_expert_traj.append(np.hstack([state, next_state]))
            if args.render:
                env.render()
            if done or num_steps >= args.max_expert_state_num:
                break

            state = next_state

        print('Episode {}\t reward: {:.2f}'.format(i_episode, reward_episode))

        if num_steps >= args.max_expert_state_num:
            break


main_loop()