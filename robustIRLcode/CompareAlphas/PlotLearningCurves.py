import pickle
import sys
sys.path.insert(0,'../src/')

from optimizers import *
# from agent import *
from environment import *
from IRLalgorithms import *
from MDPsolver import *
from utils import *
from plot import *
import argparse
import copy
import ast

## You may need to modify the path of basedir
def base_dir(alphaE, alphaL, noiseE, noiseL, seed, env_type, dim,linear,reg_opp, fix_start, iil = False ):
    if not iil:
        softQ_lr = 0.0
        folder = "../../models_gw"
        dir_path = folder +  '/fix_start'+ str(fix_start) +"/env_type" + str(env_type) +"/dim_" + str(dim)+ "/Linear"+str(linear)+"/RegOpp"+str(reg_opp)+"/alphaE_" + str(alphaE) + "_alphaL_" + str(alphaL) + \
                "/beta_0.0_beta_op_0.0/softQ_lr_" + str(softQ_lr) + "/noiseE_" + str(noiseE) + "_noiseL_" + str(noiseL) + \
                "/seed_" + str(seed) + "/0"   
    else:
        folder = "../../models_gw_IIL"
        dir_path = folder +  '/fix_start'+ str(fix_start) +"/env_type" + str(env_type) +"/dim_" + str(dim)+ "/noiseE_" + str(noiseE) + "_noiseL_" + str(noiseL) + \
                "/seed_" + str(seed) + "/0" 
    return dir_path

def load_multiple_data(lr, alphasE, alphasL, noisesE, noisesL, seed, env_type, dim, linear, reg_opp, fix_start, iil=False):
    if not iil:
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
                            results[alphaE][alphaL][noiseE][noiseL][s] = load_data(base_dir(alphaE, alphaL, noiseE, noiseL, s, env_type, dim, linear, reg_opp, fix_start, iil), lr)
    else:
        results = {}
        alphaE = alphasE[0]
        alphaL = alphasL[0]
        for noiseE in noisesE:
            results[noiseE] = {}
            for noiseL in noisesL:
                results[noiseE][noiseL] = {}
                for s in seed:
                    results[noiseE][noiseL][s] = load_data_iil(base_dir(alphaE, alphaL, noiseE, noiseL, s, env_type, dim, linear, reg_opp, fix_start, iil), lr)
    
    return results


parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=int, default=0)
parser.add_argument('--dim', type=int, default=10)
parser.add_argument('--noiseL', type=float, default = 0.0)
parser.add_argument('--alphaL', type=float, default = 0.0)
parser.add_argument('--noiseE', type=float, default = 0.0)
parser.add_argument('--include_iil', type = ast.literal_eval)
args = parser.parse_args()



lr = 0.5
lr_iil = 0.01
dim = args.dim
env_type = args.mode
fix_start = False
alphaE = "1.0"
alphasE = [alphaE]
alphasL = [str(args.alphaL)]
noisesE = [str(args.noiseE)]
noiseL = str(args.noiseL)
noisesL = [noiseL]
linear = True
reg_opp = False
seed = [0]
### CompareAlphas
compute = True
policies = []
if compute:
    results = load_multiple_data(lr, alphasE, alphasL, noisesE, noisesL, seed, env_type, dim, linear, reg_opp, fix_start)   
    for alphaL in alphasL:
        for noiseE in noisesE:
            policies.append(results[alphaE][alphaL][noiseE][noiseL][seed[0]]["player"][:5])
    if args.include_iil:
        results = load_multiple_data(lr_iil, alphasE, alphasL, noisesE, noisesL, seed, env_type, dim, linear, reg_opp, fix_start, args.include_iil)   
        for noiseE in noisesE:
            policies.append(results[noiseE][noiseL][seed[0]]["policies"][:5])
    seed = seed[0]
    env_2p = GridWorldEnvironment(env_type,dim, prop = float(noiseL))
    sol = MDPsolver(env_2p)
    Vs = {}
    sigma_Vs = {}
    if fix_start:
        starts = [ env_2p.n_states-1 for i in range(env_2p.n_states)]
    else:
        starts = [i for i in range(env_2p.n_states)]


    n_traj = len(starts)*10
    repetitions = int(n_traj/len(starts))
    
    for a, policy_story in enumerate(policies):
        Vs[a] = []
        sigma_Vs[a] = []
        for k,policy in enumerate(policy_story):
            #plot_value_and_policy(sol, policy, str(k), mode = "max_ent", show = True)
            print(policy, "POLICY")
            agent = Agent(copy.deepcopy(env_2p), policy= policy)

            partial_V = np.zeros(repetitions)

            np.random.seed(1)
            for n in range(repetitions):
                partial_V[n] = agent.evaluate_policy(int(n_traj/repetitions), starting_index = starts)
            V = np.mean(partial_V)
            sigma_V = np.std(partial_V)
            Vs[a].append(V)
            sigma_Vs[a].append(sigma_V)

    solver = MDPsolver(copy.deepcopy(env_2p))
    solver.value_iteration()
    expert = Agent(copy.deepcopy(env_2p), policy= solver.policy)
    partial_V = np.zeros(repetitions)

    np.random.seed(100)
    for n in range(repetitions):
        partial_V[n] = expert.evaluate_policy(int(n_traj/repetitions), starting_index = starts)
    V_expert = np.mean(partial_V)
    sigma_V_expert = np.std(partial_V)
    Vs["expert"] = [V_expert]
    sigma_Vs["expert"] = [sigma_V_expert]
    Vs_to_plot = []
    sigma_Vs_to_plot = []
    
    for a, policy_story in enumerate(policies):
        VP = []
        VP_sigma = []
        for i in range(len(policy_story)):

            VP.append(Vs[a].pop(0))
            VP_sigma.append(sigma_Vs[a].pop(0))
        Vs_to_plot.append(- np.array(VP))
        sigma_Vs_to_plot.append(np.array(VP_sigma))
    Vs_to_plot.append(-Vs["expert"][0]*np.ones(len(policies[0])))
    sigma_Vs_to_plot.append(sigma_Vs["expert"][0]*np.ones(len(policies[0])))
    print(Vs_to_plot, "Vs")
    print(sigma_Vs_to_plot, "sigmaVs")
    if args.include_iil:
        alphasL.append("IIL")
    alphasL.append("expert")
    plot_label = alphasL
    data = {
        "Vs_to_plot": Vs_to_plot,
        "sigma_Vs": sigma_Vs_to_plot,
        "labels": plot_label
    }
    save_result(data,   "PlotLearningCurves_env" + str(env_type) + "size" + str(dim)+"lr"+str(lr)+ \
                        "noiseL"+noiseL+"noiseE"+noiseE+\
                        "alphaE"+alphaE+"alphaL"+alphaL+"fix_start"+str(fix_start)+"include_iil" + str(args.include_iil))

else:
    noiseE = noisesE[-1]
    alphaL = alphasL[-1]
    data = load_result("PlotLearningCurves_env" + str(env_type) + "size" + str(dim)+"lr"+str(lr)+ \
                        "noiseL"+noiseL+"noiseE"+noiseE+\
                        "alphaE"+alphaE+"alphaL"+alphaL+"fix_start"+str(fix_start)+"include_iil" + str(args.include_iil))
    Vs_to_plot = data["Vs_to_plot"]
    sigma_Vs_to_plot = data["sigma_Vs"]
    plot_label = data["labels"]
###
sample_steps = np.arange(len(Vs_to_plot[0]))

plot_log_lines( list_to_plot = Vs_to_plot,
            list_name = plot_label,
            axis_label = ["IRL steps", "Total Return Delta"],
            folder = "",
            title = "learning_curve_env" + str(env_type) + "size" + str(dim)+"lr"+str(lr)+ \
                        "noiseL"+noiseL+"noiseE"+noiseE+\
                        "alphaE"+alphaE+"alphaL"+alphaL+"fix_start"+str(fix_start),
            x_axis = sample_steps)

plot_log_lines_and_ranges(  list_to_plot = Vs_to_plot,
                        list_sigmas = sigma_Vs_to_plot,
                        list_name = plot_label,
                        axis_label = ["IRL steps", "Total Return Delta"],
                        folder = "",
                        title = "learning_curve_env" + str(env_type) + "size" + str(dim)+"lr"+str(lr)+ \
                        "noiseL"+noiseL+"noiseE"+noiseE+\
                        "alphaE"+alphaE+"alphaL"+alphaL+"fix_start"+str(fix_start),
                        x_axis = sample_steps)
