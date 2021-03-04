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
def base_dir(alphaE, alphaL, noiseE, noiseL, seed, dim,linear,reg_opp, fix_start ):
    softQ_lr = 0.0
    folder = "../../models_ow"
    dir_path = folder +  '/fix_start'+ str(fix_start) +"/dim_" + str(dim)+ "/Linear"+str(linear)+"/RegOpp"+str(reg_opp)+"/alphaE_" + str(alphaE) + "_alphaL_" + str(alphaL) + \
            "/beta_0.0_beta_op_0.0/softQ_lr_" + str(softQ_lr) + "/noiseE_" + str(noiseE) + "_noiseL_" + str(noiseL) + \
            "/seed_" + str(seed) + "/0"   
    return dir_path

def load_multiple_data(lr, alphasE, alphasL, noisesE, noisesL, seed, dim, linear, reg_opp, fix_start):
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
                        results[alphaE][alphaL][noiseE][noiseL][s] = load_data(base_dir(alphaE, alphaL, noiseE, noiseL, s, dim, linear, reg_opp, fix_start), lr)
    return results
parser = argparse.ArgumentParser()
parser.add_argument('--env_seed', type=int, default=0)
parser.add_argument('--dim', type=int, default=10)
parser.add_argument('--noiseL', type=float, default = 0.0)
args = parser.parse_args()

lr = 0.15
dim = args.dim
alphaE = "1.0"
alphasE = [alphaE]
alphasL = ["1.0", "0.95", "0.9", "0.85", "0.8"]
noisesE = ["0.0", "0.05", "0.1", "0.15", "0.2"]
noiseL = str(args.noiseL)
noisesL = [noiseL]
linear = False
reg_opp = False
fix_start = False
seed = [args.env_seed]
results = load_multiple_data(lr, alphasE, alphasL, noisesE, noisesL, seed, dim, linear, reg_opp, fix_start)   

### CompareAlphas
policies = []
alpha_list = []
noiseE_list = []

for alphaL in alphasL:
    for noiseE in noisesE:
            policies.append(results[alphaE][alphaL][noiseE][noiseL][args.env_seed]["player"][-1])
            alpha_list.append(alphaL)
            noiseE_list.append(noiseE)

n_colours = 4
n_objects = 60
obj = ObjectWorldEnvironment(dim, n_objects, n_colours, seed=args.env_seed, prop=args.noiseL)

solver = MDPsolver(obj)
solver.value_iteration()
    
    
for i,item in enumerate(zip(policies, alpha_list, noiseE_list)):
    plot_value_and_policy(solver, item[0], "OWPlotPolicyDim"+str(dim)+"EnvType"+str(args.env_seed)+str(n_colours)+str(n_objects)+"Enoise"+item[2]+"Lnoise"+noiseL+"alpha"+item[1]+"fixStart"+str(fix_start), "max_ent")
