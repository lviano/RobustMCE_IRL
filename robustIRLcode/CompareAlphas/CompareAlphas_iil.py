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

## You may need to modify the path of basedir
def base_dir( noiseE, noiseL, seed, env_type, dim, fix_start ):
    softQ_lr = 0.0
    folder = "../../models_gw_IIL"
    dir_path = folder +  '/fix_start'+ str(fix_start) +"/env_type" + str(env_type) +"/dim_" + str(dim) + \
            "/noiseE_" + str(noiseE) + "_noiseL_" + str(noiseL) + \
            "/seed_" + str(seed) + "/0"   
    return dir_path

def load_multiple_data(lr,  noisesE, noisesL, seed, env_type, dim, fix_start):
    results = {}
    for noiseE in noisesE:
        results[noiseE] = {}
        for noiseL in noisesL:
            results[noiseE][noiseL] = {}
            for s in seed:
                results[noiseE][noiseL][s] = load_data_iil(base_dir( noiseE, noiseL, s, env_type, dim, fix_start), lr)
    
    return results


parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=int, default=0)
parser.add_argument('--dim', type=int, default=10)
parser.add_argument('--noiseL', type=float, default = 0.0)
args = parser.parse_args()



lr = 0.01
dim = args.dim
env_type = args.mode
fix_start = False
noisesE = ["0.0", "0.05", "0.1", "0.15", "0.2"]
noiseL = str(args.noiseL)
noisesL = [noiseL]
seed = [0]
### CompareAlphas
compute = True
policies = []

if compute:
    results = load_multiple_data(lr, noisesE, noisesL, seed, env_type, dim, fix_start)   
    seed = seed[0]
    for noiseE in noisesE:
        policies.append(results[noiseE][noiseL][seed]["policies"][-1])

    env_2p = GridWorldEnvironment(env_type,dim, prop = float(noiseL))
    sol = MDPsolver(env_2p)
    Vs = []
    sigma_Vs = []
    if fix_start:
        starts = [ env_2p.n_states-1 for i in range(env_2p.n_states)]
    else:
        starts = [i for i in range(env_2p.n_states)]

    n_traj = len(starts)*1000
    repetitions = int(n_traj/len(starts))

    for k,policy in enumerate(policies):
        #plot_value_and_policy(sol, policy, str(k), mode = "max_ent", show = True)
        agent = Agent(copy.deepcopy(env_2p), policy= policy)

        partial_V = np.zeros(repetitions)

        np.random.seed(1)
        for n in range(repetitions):
            partial_V[n] = agent.evaluate_policy(int(n_traj/repetitions), starting_index = starts)
        V = np.mean(partial_V)
        sigma_V = np.std(partial_V)
        Vs.append(V)
        sigma_Vs.append(sigma_V)

    Vs_to_plot = []
    sigma_Vs_to_plot = []
    #for i, _ in enumerate(alphasL):
    VP = []
    VP_sigma = []
    for j, _ in enumerate(noisesE):
        VP.append(Vs.pop(0))
        VP_sigma.append(sigma_Vs.pop(0))
    Vs_to_plot.append(np.array(VP))
    sigma_Vs_to_plot.append(np.array(VP_sigma))
    plot_label = ["IIL"]
    data = {
        "Vs_to_plot": Vs_to_plot,
        "sigma_Vs": sigma_Vs_to_plot,
        "labels": plot_label
    }
    save_result(data, "IILCompareAlphas_env" + str(env_type) + "size" + str(dim)+"lr"+str(lr)+noiseL + "fix_start"+str(fix_start))

else:
    data = load_result( "IILCompareAlphas_env" + str(env_type) + "size" + str(dim)+"lr"+str(lr)+noiseL + "fix_start"+str(fix_start))
    Vs_to_plot = data["Vs_to_plot"]
    sigma_Vs_to_plot = data["sigma_Vs"]
    plot_label = data["labels"]
###
for i,enoise in enumerate(noisesE):
    noisesE[i] = float(enoise)

plot_lines( list_to_plot = Vs_to_plot,
            list_name = plot_label,
            axis_label = ["Expert Noise", "Total Return "],
            folder = "",
            title = "IILCompare Alphas Env "+str(env_type)+" noise L "+ noiseL+"dim"+str(dim)+"fix_start"+str(fix_start),
            x_axis = noisesE)

plot_lines_and_ranges(  list_to_plot = Vs_to_plot,
                        list_sigmas = sigma_Vs_to_plot,
                        list_name = plot_label,
                        axis_label = ["Expert Noise", "Total Return"],
                        folder = "",
                        title = "IILCompare Alphas Env "+str(env_type)+" noise L "+ noiseL+"dim"+str(dim)+"fix_start"+str(fix_start),
                        x_axis = noisesE)
