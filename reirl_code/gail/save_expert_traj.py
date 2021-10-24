import argparse
import gym
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
parser.add_argument('--model-path', metavar='G',
                    help='name of the expert model')
parser.add_argument('--render', action='store_true', default=False,
                    help='render the environment')
parser.add_argument('--seed', type=int, default=1, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--max-expert-state-num', type=int, default=50000, metavar='N',
                    help='maximal number of main iterations (default: 50000)')
parser.add_argument('--noiseE', type=float, default=0.0, metavar='G')
parser.add_argument('--unif-noise', type=float, default=0, metavar='G')
parser.add_argument('--grid-type', type=int, default=None, metavar='N')
parser.add_argument('--mass-mul', type=float, default=1.0, metavar='G',
                    help="Multiplier for CartPole and Acrobot masses")
parser.add_argument('--len-mul', type=float, default=1.0, metavar='G',
                    help="Multiplier for CartPole and Acrobot lengths")
args = parser.parse_args()

dtype = torch.float64
torch.set_default_dtype(dtype)
if args.env_name == "gridworld-v0" or args.env_name == "ContinuousGridworld-v0" or args.env_name=="GaussianGridworld-v0":
    env = gym.make(args.env_name, prop = args.noiseE, env_type = args.grid_type)
    subfolder = "env"+str(args.env_name)+"type"+str(args.grid_type)+"noiseE"+str(args.noiseE)
    if not os.path.isdir(assets_dir(subfolder+"/expert_traj")):
        os.makedirs(assets_dir(subfolder+"/expert_traj"))
elif args.env_name == "CartPole-v1" or args.env_name == "Acrobot-v1":
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
elif args.env_name == "HalfCheetah-v2" or args.env_name == "Walker2d-v2" or args.env_name == "Hopper-v2" or args.env_name == "Swimmer-v2" or args.env_name == "InvertedPendulum-v2" or args.env_name == "InvertedDoublePendulum-v2" or args.env_name == "Ant-v2":
    env = gym.make(args.env_name)
    env.env.model.body_mass[:] *= args.mass_mul
    subfolder = "env"+ args.env_name + "mass" + str(args.mass_mul)
    if not os.path.isdir(assets_dir(subfolder+"/expert_traj")):
        os.makedirs(assets_dir(subfolder+"/expert_traj"))
else:
    env = gym.make(args.env_name)
    subfolder = "env" + args.env_name +"unif_noise"+str(args.unif_noise)
    if not os.path.isdir(assets_dir(subfolder+"/expert_traj")):
        os.makedirs(assets_dir(subfolder+"/expert_traj"))
env.seed(args.seed)
torch.manual_seed(args.seed)
is_disc_action = len(env.action_space.shape) == 0
state_dim = env.observation_space.shape[0]

policy_net, _ = pickle.load(open(args.model_path, "rb"))
running_state= lambda x: x


def main_loop():

    num_steps = 0
    expert_traj = []
    state_only_expert_traj = []
    for i_episode in count():

        state = env.reset()
        state = running_state(state)
        print(state.shape)
        reward_episode = 0
        states_actions = []
        states_next_states = []
        for t in range(10000):
            state_var = tensor(state).unsqueeze(0).to(dtype)
            # choose mean action
            if not is_disc_action:
                action = policy_net(state_var)[0][0].detach().numpy()
            else:
                action = policy_net.select_action(state_var)[0].numpy()
            # action = policy_net.select_action(state_var)[0].cpu().numpy()
            action = int(action) if is_disc_action else action.astype(np.float64)
            if not is_disc_action:
                z = np.random.uniform()
                if args.unif_noise > z:
                    action = np.random.uniform(low = - 1.0, high = 1.0, size = action.shape)
            next_state, reward, done, _ = env.step(action)
            next_state = running_state(next_state)
            reward_episode += reward
            num_steps += 1

            states_actions.append(np.hstack([state, action]))
            states_next_states.append(np.hstack([state, next_state]))

            if args.render:
                env.render()
            if done or num_steps >= args.max_expert_state_num:
                #if reward_episode > 0:
                states_actions.append(np.hstack([next_state, action]))
                states_next_states.append(np.hstack([next_state, next_state]))
                expert_traj += states_actions
                state_only_expert_traj += states_next_states
                break

            state = next_state

        print('Episode {}\t reward: {:.2f}'.format(i_episode, reward_episode))

        if num_steps >= args.max_expert_state_num:
            break
        #import matplotlib.pyplot as plt
        #plt.scatter(np.array(expert_traj)[:,0], np.array(expert_traj)[:,1])
        #plt.savefig("Episode"+str(i_episode))
    return expert_traj, state_only_expert_traj


expert_traj, state_only_expert_traj = main_loop()
expert_traj = np.stack(expert_traj)
state_only_expert_traj = np.stack(state_only_expert_traj)
pickle.dump((expert_traj), open(os.path.join(assets_dir(subfolder), 'expert_traj/{}_expert_traj.p'.format(args.env_name)), 'wb'))
pickle.dump((state_only_expert_traj), open(os.path.join(assets_dir(subfolder), 'expert_traj/{}_state_only_expert_traj.p'.format(args.env_name)), 'wb'))
