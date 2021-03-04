import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import copy
import pickle

import argparse
import os
import ast

import sys
sys.path.insert(0, 'src/')

from optimizers import *
from environment import *
from IRLalgorithms import *
from MDPsolver import *
from utils import *
from RewardNet import FourLayersNet, RewardNet
from plot import plot_objectworld, plot_value_and_policy, plot_on_grid
from IILsolver import *

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--dim', type=int, default=10)
parser.add_argument('--mode', type=int, default=8)
parser.add_argument('--seed', type=int, default=10)
parser.add_argument('--noiseE', type=float, default=0.0)
parser.add_argument('--noiseL', type=float, default=0.0)
parser.add_argument('--linear', type = ast.literal_eval)
parser.add_argument('--fix_start', type = ast.literal_eval)
args = parser.parse_args()

base_dir = os.getcwd() + '/../models_ow_IIL/'
base_dir += '/fix_start'+ str(args.fix_start) + '/dim_'+ str(args.dim) + \
            '/noiseE_' + str(args.noiseE) + '_noiseL_' + str(args.noiseL) + '/seed_' + str(args.seed) + '/'

run_number = 0
while os.path.exists(base_dir + str(run_number)):
    run_number += 1
base_dir = base_dir + str(run_number) + '/'

os.makedirs(base_dir)

# setup_seed(20) ## seed for grid world
setup_seed(args.seed)

fix_horizon = False

dim = args.dim
n_objects = 6
n_colours = 2
if fix_horizon:
    ObjectWorld = ObjectWorldEnvironment(dim, n_objects, n_colours, seed=args.seed, prop=args.noiseE)
else:
    ObjectWorld = Inf_Horizon_ObjectWorldEnvironment(dim, n_objects, n_colours, seed=args.seed, prop=args.noiseE)

solver = MDPsolver(ObjectWorld)
#plot_objectworld(ObjectWorld, args.dim, title = "PPP32"+str(args.seed), log_color = False, show = True)

if fix_horizon:
    player, adv = solver.soft_2player_value_iteration_fixed_horizon(alpha = 1.0)
    solver.value_iteration_fixed_horizon()
else:
    player, adv = solver.soft_2player_value_iteration(alpha = 1.0)
    solver.value_iteration()

player = randomize_optimal_policy(solver)
#print(player.shape)
#plot_value_and_policy(solver, player, "title", mode = "max_ent", show = True)

n_traj = 5000
teacher = Agent(ObjectWorld, policy = player)
demon_states, demon_actions, mus_demons, svf_demons, demon_rewards = teacher.collect_trajectories(n_traj, limit=30)
mean_svf_demons = np.mean(svf_demons, axis=0)

p_initial = compute_initial_probabilities(demon_states, ObjectWorld)
print(demon_states)
solver.policy = np.copy(player)
if fix_horizon:
    if args.fix_start:
        mu_teacher = solver.mu_policy_fixed_horizon(solver.policy,stochastic=True, emp_p_in=p_initial)
    else:
        mu_teacher = solver.mu_policy_fixed_horizon(solver.policy,stochastic=True)
else:
    if args.fix_start:
        mu_teacher = solver.mu_policy(solver.policy,stochastic=True, emp_p_in=p_initial)
    else:
        mu_teacher = solver.mu_policy(solver.policy,stochastic=True)

if fix_horizon:
    learned_ObjectWorld = ObjectWorldEnvironment(dim, n_objects, n_colours, seed=args.seed, prop=args.noiseL)
else:
    learned_ObjectWorld = Inf_Horizon_ObjectWorldEnvironment(dim, n_objects, n_colours, seed=args.seed, prop=args.noiseL)
learned_Solver = MDPsolver(learned_ObjectWorld)
#IRL = IRLsolver(learned_Solver, mu_teacher = mean_svf_demons)

#plot_on_grid(mean_svf_demons, dim, show = True)

IIL = trajectoryIILsolver(learned_Solver, teacher_states=demon_states, no_one_hot=True)
IIL.i2l(base_dir,  lr = args.lr, no_one_hot = True)