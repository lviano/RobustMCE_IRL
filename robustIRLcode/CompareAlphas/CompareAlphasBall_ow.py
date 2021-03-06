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
def base_dir(alphaE, alphaL, noiseE, noiseL, seed, dim,linear,reg_opp, fix_start ):
    softQ_lr = 0.0
    folder = "../../models_ow"
    dir_path = folder +  '/fix_start'+ str(fix_start) +"/dim_" + str(dim)+ "/Linear"+str(linear)+"/RegOpp"+str(reg_opp)+"/alphaE_" + str(alphaE) + "_alphaL_" + str(alphaL) + \
            "/beta_0.0_beta_op_0.0/softQ_lr_" + str(softQ_lr) + "/noiseE_" + str(noiseE) + "_noiseL_" + str(noiseL) + \
            "/seed_" + str(seed) + "/0"   
    return dir_path

def load_multiple_data(lr, alphasE, alphasL, noisesE, noisesL, seed_env, dim, linear, reg_opp, fix_start):
    results = {}
    for alphaE in alphasE:
        results[alphaE] = {}
        for alphaL in alphasL:
            results[alphaE][alphaL] = {}
            for noiseE in noisesE:
                results[alphaE][alphaL][noiseE] = {}
                for noiseL in noisesL:
                    results[alphaE][alphaL][noiseE][noiseL] = {}
                    for s in seed_env:
                        results[alphaE][alphaL][noiseE][noiseL][s] = load_data(base_dir(alphaE, alphaL, noiseE, noiseL, s, dim, linear, reg_opp, fix_start), lr)
    return results
parser = argparse.ArgumentParser()
parser.add_argument('--env_seed', type=int, default=0)
parser.add_argument('--dim', type=int, default=10)
parser.add_argument('--noiseL', type=float, default = 0.0)
args = parser.parse_args()

lr = 0.001
dim = args.dim
alphaE = "1.0"
alphasE = [alphaE]
alphasL = ["1.0", "0.95", "0.9", "0.85", "0.8"]
noisesE = ["0.1", "0.15", "0.2", "0.25","0.3"]
noiseL = str(args.noiseL)
noisesL = [noiseL]
linear = False
reg_opp = False
fix_start = False
compute = False
### CompareAlphas
if compute:
    results = load_multiple_data(lr, alphasE, alphasL, noisesE, noisesL, [args.env_seed], dim, linear, reg_opp, fix_start)   
    policies = []
    for alphaL in alphasL:
        for noiseE in noisesE:
                policies.append(results[alphaE][alphaL][noiseE][noiseL][args.env_seed]["player"][-1])


    n_objects = 6
    n_colours = 2

    ObjectWorld = ObjectWorldEnvironment(dim, n_objects, n_colours, seed=args.env_seed, prop=args.noiseL)
    solver = MDPsolver(ObjectWorld)
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
            partial_V[n] = agent.evaluate_policy(int(n_traj/repetitions),  limit = args.dim*2, starting_index = starts)
        V = np.mean(partial_V)
        sigma_V = np.std(partial_V)
        Vs.append(V)
        sigma_Vs.append(sigma_V)

    solver = MDPsolver(copy.deepcopy(ObjectWorld))
    solver.value_iteration()
    expert = Agent(copy.deepcopy(ObjectWorld), policy= solver.policy)
    partial_V = np.zeros(repetitions)

    np.random.seed(100)
    for n in range(repetitions):
        partial_V[n] = expert.evaluate_policy(int(n_traj/repetitions), limit = args.dim*2, starting_index = starts)
    V_expert = np.mean(partial_V)
    sigma_V_expert = np.std(partial_V)
    Vs.append(V_expert)
    sigma_Vs.append(sigma_V_expert)

    Vs_to_plot = []
    sigma_Vs_to_plot = []
    for i, _ in enumerate(alphasL):
        VP = []
        VP_sigma = []
        for j, _ in enumerate(noisesE):
            VP.append(Vs.pop(0))
            VP_sigma.append(sigma_Vs.pop(0))
        Vs_to_plot.append(np.array(VP))
        sigma_Vs_to_plot.append(np.array(VP_sigma))
    Vs_to_plot.append(Vs[0]*np.ones(len(noisesE)))
    sigma_Vs_to_plot.append(sigma_Vs[0]*np.ones(len(noisesE)))
    alphasL.append("expert")
    plot_label = alphasL
    data = {
        "Vs_to_plot": Vs_to_plot,
        "sigma_Vs": sigma_Vs_to_plot,
        "labels": plot_label
    }
    save_result(data, "OWCompareAlphas_env" + str(args.env_seed) + "size" + str(dim)+"lr"+str(lr)+noiseL +"alphaE"+alphaE+"fix_start"+str(fix_start))

else:
    data = load_result( "fixOWCompareAlphas_env" + str(args.env_seed) + "size" + str(dim)+"lr"+str(lr)+noiseL +"alphaE"+alphaE+"fix_start"+str(fix_start)+"run0")
    Vs_to_plot = data["Vs_to_plot"]
    sigma_Vs_to_plot = data["sigma_Vs"]
    plot_label = data["labels"]
###
for i,enoise in enumerate(noisesE):
    noisesE[i] = float(enoise) - float(noiseL)

Vs_to_plot.pop(5) 
plot_label.pop(5)
sigma_Vs_to_plot.pop(5)

Vs_to_plot_short = []
plot_label_short = []
sigma_Vs_to_plot_short = []

for i, item in enumerate(zip(Vs_to_plot, plot_label, sigma_Vs_to_plot)):

    Vs_to_plot_short.append(item[0][2:])
    sigma_Vs_to_plot_short.append(item[2][2:])



plot_lines( list_to_plot = Vs_to_plot_short,
            list_name = plot_label,
            axis_label = ["Noise Difference", "Total Return "],
            folder = "",
            title = "BallOWNo0.6ConvergedCompare Alphas Env "+ str(args.env_seed) +" noise L "+ noiseL+"dim"+str(dim)+"alphaE"+alphaE+"fix_start"+str(fix_start),
            x_axis = noisesE)

plot_lines_and_ranges(  list_to_plot = Vs_to_plot_short,
                        list_sigmas = sigma_Vs_to_plot_short,
                        list_name = plot_label,
                        axis_label = ["Noise Difference", "Total Return "],
                        folder = "",
                        title = "BallOWNo0.6ConvergedCompare Alphas Env "+ str(args.env_seed) +" noise L "+ noiseL+"dim"+str(dim)+"alphaE"+alphaE+"fix_start"+str(fix_start),
                        x_axis = noisesE)
