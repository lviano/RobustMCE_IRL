import pickle
import sys
sys.path.insert(0,'../src/')

from optimizers import *
from environment import *
from IRLalgorithms import *
from MDPsolver import *
from utils import *
from plot import *
import argparse
import copy


## You may need to modify the path of basedir
def base_dir(alphaE, alphaL, noiseE, noiseL, seed, dim, linear, reg_opp, fix_start, fix_horizon):
    softQ_lr = 0.0
    folder = "../../models_ow"
    dir_path = folder +  '/fix_horizon' + str(fix_horizon) + '/fix_start' + str(fix_start) + "/dim_" + str(dim)+ "/Linear" \
            + str(linear) + "/RegOpp" + str(reg_opp) + "/alphaE_" + str(alphaE) + "_alphaL_" + str(alphaL) + \
            "/beta_0.0_beta_op_0.0/softQ_lr_" + str(softQ_lr) + "/noiseE_" + str(noiseE) + "_noiseL_" + str(noiseL) + \
            "/seed_" + str(seed) + "/0"
    return dir_path

def load_multiple_data(lr, alphasE, alphasL, noisesE, noisesL, seed, dim, linear, reg_opp, fix_start, fix_horizon):
    results = {}
    for alphaE in alphasE:
        results[alphaE] = {}
        for alphaL in alphasL:
            results[alphaE][alphaL] = {}
            for noiseE in noisesE:
                results[alphaE][alphaL][noiseE] = {}
                for noiseL in noisesL:
                    results[alphaE][alphaL][noiseE][noiseL] = {}
                    for s in seed:
                        results[alphaE][alphaL][noiseE][noiseL][s] = load_data(base_dir(alphaE, alphaL, noiseE, noiseL, s, dim, linear, reg_opp, fix_start, fix_horizon), lr)
    return results

def policy_ow_loader(dim, env_seed, noiseL_inp, fix_horizon):
    lr = 0.001
    fix_start = False

    alphaE = "1.0"
    alphasE = [alphaE]
    alphasL = ["1.0", "0.95", "0.9", "0.85", "0.8"]#, "0.6"]
    noisesE = ["0.0", "0.05", "0.1", "0.15", "0.2", "0.25", "0.3"]
    noiseL = str(noiseL_inp)
    noisesL = [noiseL]
    linear = False
    reg_opp = False
    seed = [env_seed]
    policies = []
    alpha_list = []
    noiseE_list = []

    results = load_multiple_data(lr, alphasE, alphasL, noisesE, noisesL, seed, dim, linear, reg_opp, fix_start, fix_horizon)

    for alphaL in alphasL:
        for noiseE in noisesE:
                policies.append(results[alphaE][alphaL][noiseE][noiseL][env_seed]["player"][-1])
                alpha_list.append(alphaL)
                noiseE_list.append(noiseE)

    n_colours = 2
    n_objects = 6
    obj = ObjectWorldEnvironment(dim, n_objects, n_colours, seed=env_seed, prop=noiseL_inp)

    solver = MDPsolver(obj)
    solver.value_iteration()

    for _,item in enumerate(zip(policies, alpha_list, noiseE_list)):
        print("alpha: " + str(item[1]) + " noiseE: " + str(item[2]))
        plot_value_and_policy(solver, item[0], "","max_ent", show = True)
