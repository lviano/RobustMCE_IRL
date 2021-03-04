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
def base_dir(alphaE, alphaL, noiseE, noiseL, seed, env_type, dim,linear,reg_opp, fix_start, folder, run ):
    softQ_lr = 0.0
    dir_path = folder +  '/fix_start'+ str(fix_start) +"/env_type" + str(env_type) +"/dim_" + str(dim)+ "/Linear"+str(linear)+"/RegOpp"+str(reg_opp)+"/alphaE_" + str(alphaE) + "_alphaL_" + str(alphaL) + \
            "/beta_0.0_beta_op_0.0/softQ_lr_" + str(softQ_lr) + "/noiseE_" + str(noiseE) + "_noiseL_" + str(noiseL) + \
            "/seed_" + str(seed) + "/"+str(run)   
    return dir_path

def load_multiple_data(lr, alphasE, alphasL, noisesE, noisesL, seed, env_type, dim, linear, reg_opp, fix_start, folder, run):
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
                        results[alphaE][alphaL][noiseE][noiseL][s] = load_data(base_dir(alphaE, alphaL, noiseE, noiseL, s, env_type, dim, linear, reg_opp, fix_start, folder, run), lr)
    return results

def policy_loader(dim, env_type, noiseL_inp, folder = "../../models_gw", alphaE = "1.0", fix_start = False, run = 0):
    if fix_start:
        lr = 0.15
    else:
        lr = 0.5
    alphasE = [alphaE]
    alphasL = ["1.0", "0.95", "0.9", "0.85", "0.8", "0.6"]
    noisesE = ["0.0", "0.05", "0.1", "0.15", "0.2"]
    noiseL = str(noiseL_inp)
    noisesL = [noiseL]
    linear = True
    reg_opp = False
    seed = [0]
    policies = []
    alpha_list = []
    noiseE_list = []
    

    
    results = load_multiple_data(lr, alphasE, alphasL, noisesE, noisesL, seed, env_type, dim, linear, reg_opp, fix_start, folder, run)   
    seed = 0
    for alphaL in alphasL:
        for noiseE in noisesE:
                policies.append(results[alphaE][alphaL][noiseE][noiseL][seed]["player"][-1])
                alpha_list.append(alphaL)
                noiseE_list.append(noiseE)
    env = GridWorldEnvironment(env_type, dim, noiseL_inp)
    solver = MDPsolver(env)
    solver.value_iteration()
    for i,item in enumerate(zip(policies, alpha_list, noiseE_list)):
        print("alpha: " + str(item[1]) + " noiseE: " + str(item[2]))
        plot_value_and_policy(solver, item[0], "","max_ent", show = True)
