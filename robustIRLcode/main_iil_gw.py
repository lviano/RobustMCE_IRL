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
from agent import Agent
from environment import *
from IRLalgorithms import IRLsolver
from IILsolver import *
from MDPsolver import MDPsolver
from utils import *
from RewardNet import RewardNet, adjust_learning_rate
from plot import plot_on_grid, plot_reward


parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--dim', type=int, default=5)
parser.add_argument('--mode', type=int, default=8)
parser.add_argument('--seed', type=int, default=10)
parser.add_argument('--noiseE', type=float, default=0)
parser.add_argument('--noiseL', type=float, default=0)
parser.add_argument('--fix_start', type = ast.literal_eval)

args = parser.parse_args()

base_dir = os.getcwd() + '/../models_gw_IIL/'
base_dir +=  '/fix_start'+ str(args.fix_start) + "/env_type" + str(args.mode) +'/dim_'+ str(args.dim) + '/noiseE_' + str(args.noiseE) + '_noiseL_' + str(args.noiseL) + '/seed_' + str(args.seed) + '/'

run_number = 0
while os.path.exists(base_dir + str(run_number)):
    run_number += 1
base_dir = base_dir + str(run_number) + '/'

os.makedirs(base_dir)

setup_seed(args.seed)
fixed_start = args.fix_start
mode = args.mode
dim = args.dim
gridworld = GridWorldEnvironment(mode, dim, prop=args.noiseE)
solver = MDPsolver(gridworld)


player, adv = solver.soft_2player_value_iteration(alpha = 1.0)

#### Uncomment for empirical feature demonstration
n_traj = 1000
teacher = Agent(gridworld, policy = player)
if fixed_start:
    start = dim*dim - 1
else:
    start = None
demon_states, demon_actions, mus_demons, svf_demons, demon_rewards = teacher.collect_trajectories(n_traj, limit = 1000, start = start)
mean_mu_demons = np.mean(mus_demons, axis=0)
expert_solver = copy.deepcopy(solver)
solver.policy = np.copy(player)
rand_pi = player
p_initial = compute_initial_probabilities(demon_states, gridworld)

if fixed_start:
    mu_teacher = solver.mu_policy(solver.policy,stochastic=True, emp_p_in=p_initial)
else:
    mu_teacher = solver.mu_policy(solver.policy,stochastic=True)

learned_GridWorld = GridWorldEnvironment(mode, dim, prop=args.noiseL)
learned_GridWorld.r = np.zeros((learned_GridWorld.n_states, learned_GridWorld.n_actions))
learned_Solver = MDPsolver(learned_GridWorld)
#Indirect Imitation Learning
#IIL = IILsolver(learned_Solver, mu_teacher = mu_teacher)
IIL = trajectoryIILsolver(learned_Solver, teacher_states=demon_states)
IIL.i2l(base_dir,  lr = args.lr)
