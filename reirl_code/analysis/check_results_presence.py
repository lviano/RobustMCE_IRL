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

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', metavar='G',
                    help='name of the environment to run')
parser.add_argument("--alg", nargs='+', default=["0"])
parser.add_argument("--alpha", nargs='+', default=["0"])
parser.add_argument("--seeds", nargs='+', default=["0"])
parser.add_argument("--noiseE", nargs='+', default=["0"],
                    help='expert noise')
parser.add_argument("--mass-muls", nargs='+', default=["0"],
                    help='expert masses')
parser.add_argument("--len-muls", nargs='+', default=["0"],
                    help='expert lengths')
parser.add_argument("--render", default=False, action='store_true')
parser.add_argument("--no-compute", default=False, action='store_true')
parser.add_argument("--var-mass", default=False, action='store_true')
parser.add_argument("--var-len", default=False, action='store_true')
parser.add_argument("--vi-expert", default=False, action='store_true')
parser.add_argument("--best", default=False, action='store_true')
parser.add_argument('--exp-type', type=str, default="mismatch", metavar='N',
                    help="experiment type: noise or mismatch")
parser.add_argument("--friction", default=False, action='store_true')
parser.add_argument("--max-steps", type=int, default=5000)
args = parser.parse_args()
gail_done = False
expert_done = False

for alg in args.alg:
    for alpha in args.alpha:
        for mass_mulL in args.mass_muls:
            for seed in args.seeds:
                if not args.friction:
                    data_subfolder = "env" + args.env_name + "massL" + str(mass_mulL) + "massE1.0"
                else:
                    data_subfolder = "env" + args.env_name + "frictionL" + str(mass_mulL) + "frictionE1.0"
                if os.path.exists(os.path.join(assets_dir(data_subfolder),'learned_models/{}_{}_best_{}.p'.format(args.env_name + str(seed), alg, alpha))):
                    print(seed, alg, alpha, mass_mulL, "is present")
                else:
                    print(seed, alg, alpha, mass_mulL, "is missing")
