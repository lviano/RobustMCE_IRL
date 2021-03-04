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
def base_dir( noiseE, noiseL, seed, dim, fix_start ):
    softQ_lr = 0.0
    folder = "../../models_ow_IIL"
    dir_path = folder +  '/fix_start'+ str(fix_start) + "/dim_" + str(dim) + \
            "/noiseE_" + str(noiseE) + "_noiseL_" + str(noiseL) + \
            "/seed_" + str(seed) + "/0"   
    return dir_path

def load_multiple_data(lr,  noisesE, noisesL, seed, dim, fix_start):
    results = {}
    for noiseE in noisesE:
        results[noiseE] = {}
        for noiseL in noisesL:
            results[noiseE][noiseL] = {}
            results[noiseE][noiseL][seed] = load_data_iil(base_dir( noiseE, noiseL, seed, dim, fix_start), lr)
    
    return results


parser = argparse.ArgumentParser()
parser.add_argument('--env_seed', type=int, default=10)
parser.add_argument('--dim', type=int, default=10)
parser.add_argument('--noiseL', type=float, default = 0.0)
args = parser.parse_args()



lr = 0.01
dim = args.dim

fix_start = False
noisesE = ["0.0", "0.05", "0.1", "0.15", "0.2"]
noiseL = str(args.noiseL)
noisesL = [noiseL]
### CompareAlphas
compute = True
policies = []

if compute:
    results = load_multiple_data(lr, noisesE, noisesL, args.env_seed, dim, fix_start)   
    for noiseE in noisesE:
        policies.append(results[noiseE][noiseL][args.env_seed]["policies"][-1])

    n_objects = 6
    n_colours = 2

    ObjectWorld = Inf_Horizon_ObjectWorldEnvironment(dim, n_objects, n_colours, seed=args.env_seed, prop=args.noiseL)
    sol = MDPsolver(ObjectWorld)
    Vs = []
    sigma_Vs = []
    if fix_start:
        starts = [ ObjectWorld.n_states-1 for i in range(ObjectWorld.n_states)]
    else:
        starts = [i for i in range(ObjectWorld.n_states)]

    n_traj = len(starts)*1000
    repetitions = int(n_traj/len(starts))

    for k,policy in enumerate(policies):
        #plot_value_and_policy(sol, policy, str(k), mode = "max_ent", show = True)
        agent = Agent(copy.deepcopy(ObjectWorld), policy= policy)

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
    save_result(data, "IIL_OW_CompareAlphas_env" + str(args.env_seed) + "size" + str(dim)+"lr"+str(lr)+noiseL + "fix_start"+str(fix_start))

else:
    data = load_result( "IIL_OW_CompareAlphas_env" + str(args.env_seed) + "size" + str(dim)+"lr"+str(lr)+noiseL + "fix_start"+str(fix_start))
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
            title = "IIL_OW_Compare Alphas Env "+str(args.env_seed)+" noise L "+ noiseL+"dim"+str(dim)+"fix_start"+str(fix_start),
            x_axis = noisesE)

plot_lines_and_ranges(  list_to_plot = Vs_to_plot,
                        list_sigmas = sigma_Vs_to_plot,
                        list_name = plot_label,
                        axis_label = ["Expert Noise", "Total Return"],
                        folder = "",
                        title = "IIL_OW_Compare Alphas Env "+str(args.env_seed)+" noise L "+ noiseL+"dim"+str(dim)+"fix_start"+str(fix_start),
                        x_axis = noisesE)
