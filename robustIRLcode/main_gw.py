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
from MDPsolver import MDPsolver
from utils import *
from RewardNet import RewardNet, adjust_learning_rate
from plot import plot_on_grid, plot_reward


parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--softQ_lr', type=float, default=0.2)
parser.add_argument('--dim', type=int, default=10)
parser.add_argument('--mode', type=int, default=8)
parser.add_argument('--seed', type=int, default=10)
parser.add_argument('--beta', type=float, default=0.3)
parser.add_argument('--beta_op', type=float, default=0.3)
parser.add_argument('--alphaE', type=float, default=0.9)
parser.add_argument('--alphaL', type=float, default=0.9)
parser.add_argument('--noiseE', type=float, default=0)
parser.add_argument('--noiseL', type=float, default=0)
parser.add_argument('--linear', type = ast.literal_eval)
parser.add_argument('--reg_opp', type = ast.literal_eval)
parser.add_argument('--fix_start', type = ast.literal_eval)
parser.add_argument("--fix_horizon", type=ast.literal_eval)

args = parser.parse_args()

base_dir = os.getcwd() + '/../models_gw/'
base_dir +=  '/fix_start'+ str(args.fix_start) + "/env_type" + str(args.mode) +'/dim_'+ str(args.dim) + '/Linear'+ str(args.linear) + '/RegOpp'+ str(args.reg_opp) + '/alphaE_'+ str(args.alphaE) + "_alphaL_" + str(args.alphaL) + '/beta_' + str(args.beta) + '_beta_op_' + str(args.beta_op) + \
            '/softQ_lr_'+ str(args.softQ_lr) + '/noiseE_' + str(args.noiseE) + '_noiseL_' + str(args.noiseL) + '/seed_' + str(args.seed) + '/'

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


if args.reg_opp:
    player, adv = solver.two_players_soft_Q(alpha = args.alphaE, beta = args.beta, beta_op = args.beta_op, n_episodes = 1000, lr = args.softQ_lr, reuseQ=False)
else:
    player, adv = solver.soft_2player_value_iteration(alpha = args.alphaE) 

#### Uncomment for empirical feature demonstration
n_traj = 1000
teacher = Agent(gridworld, policy = player)
if fixed_start:
    start = dim*dim - 1
else:
    start = None
demon_states, demon_actions, mus_demons, svf_demons, demon_rewards = teacher.collect_trajectories(n_traj, limit = 1000, start = start)
#mean_mu_demons = np.mean(mus_demons, axis=0)

expert_solver = copy.deepcopy(solver)
solver.policy = np.copy(player)
rand_pi = player
p_initial = compute_initial_probabilities(demon_states, gridworld)

if fixed_start:
    mu_teacher = solver.mu_policy(solver.policy,stochastic=True, emp_p_in=p_initial)
else:
    mu_teacher = solver.mu_policy(solver.policy,stochastic=True)

learned_GridWorld = GridWorldEnvironment(mode, dim, prop=args.noiseL)

learned_Solver = MDPsolver(learned_GridWorld)
IRL = IRLsolver(learned_Solver, mu_teacher = mu_teacher)

if not args.linear:
    width = int(np.sqrt(IRL.solver.env.n_states))
    num_features = IRL.solver.env.features.shape[0]
    reward_nn = RewardNet(num_features).float()
    optimizer = optim.Adam(reward_nn.parameters(), lr=args.lr)
    policies, player, adv,  reward, err, v = IRL.deep_max_ent(base_dir, optimizer, reward_nn, p_initial, args.beta, args.beta_op, args.alphaL, args.softQ_lr, IRL_lr = args.lr, reg_opp = args.reg_opp)

else:
    policies, player, adv,  reward, err, v = IRL.max_ent_irl(base_dir, lr=args.lr,emp_p_in = p_initial, beta = args.beta, beta_op = args.beta_op, softQ_lr = args.softQ_lr, two_players=True, alpha = args.alphaL,  reg_opp = args.reg_opp)
             
