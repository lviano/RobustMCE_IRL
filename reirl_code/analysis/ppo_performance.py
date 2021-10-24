import pickle
import sys
import argparse
import copy
import ast
import os
import gym
import gym_simple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from itertools import count
from utils import *
from utils import plot
from gail.value_iteration_gym import PolicyNet, value_iteration
from evaluate_loop import evaluate_loop
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', metavar='G',
                    help='name of the environment to run')
parser.add_argument("--friction", default=False, action='store_true')
parser.add_argument("--render", default=False, action='store_true')
parser.add_argument("--max-steps", type=int, default=5000)
args = parser.parse_args()

dtype = torch.float64
torch.set_default_dtype(dtype)

to_plot = []
to_plot_std = []
if args.friction:
    subfolder = "env" + str(args.env_name) + "friction" + str(args.friction)
else:
    subfolder = "env" + str(args.env_name)

if not os.path.isdir(assets_dir(subfolder)):
    os.makedirs(assets_dir(subfolder))

torch.manual_seed(0)
env = gym.make(args.env_name)
is_disc_action = len(env.action_space.shape) == 0
state_dim = env.observation_space.shape[0]

if not os.path.isdir("../plot/" + subfolder):
    os.makedirs("../plot/" + subfolder)

gail_done = False
expert_done = False
to_plot = []
to_plot_std = []
for mass_mulL in ["0.25", "0.5", "0.75", "1.0", "2.0", "3.0", "4.0"]:
    env = gym.make(args.env_name)
    env.seed(0)
    if not args.friction:
        env.env.model.body_mass[:] *= float(mass_mulL)
        model_subfolder = "env" + str(args.env_name) + "friction" + mass_mulL
    else:
        env.env.model.geom_friction[:] *= float(mass_mulL)
        model_subfolder = "env" + str(args.env_name) + "mass" + mass_mulL


    policy, _, running_state = pickle.load(open(assets_dir(model_subfolder) + "/learned_models/"+args.env_name+"_ppo.p", "rb"))
    means = []
    for _ in range(2):
        mean_reward = evaluate_loop(policy, running_state, False, env, is_disc_action, args)
        means.append(mean_reward)

    to_plot.append(np.mean(means))
    to_plot_std.append(np.std(means))

pickle.dump((to_plot, to_plot_std), open("../plot/" + subfolder + "/PPODataCompareAlphas"
                                                + args.env_name + "friction" + str(
    args.friction) + ".p", 'wb'))
