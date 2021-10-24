import pickle
import sys
import argparse
import copy
import ast
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import *
from utils import plot

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', metavar='G',
                    help='name of the environment to run')
parser.add_argument("--alg", nargs='+', default=["0"])
parser.add_argument("--alpha", nargs='+', default=["0"])
parser.add_argument("--seeds", nargs='+', default=["0"])
parser.add_argument("--noiseE", metavar='G',
                    help='expert noise')
parser.add_argument("--noiseL", metavar='G',
                    help='learner noise')
parser.add_argument("--grid-type", metavar='G',
                    help='learner noise')
parser.add_argument("--friction", default=False, action='store_true')
parser.add_argument('--mass-mulL', type=float, default=1.0, metavar='G',
                    help="Multiplier for CartPole and Acrobot masses for learner environment")
parser.add_argument('--len-mulL', type=float, default=1.0, metavar='G',
                    help="Multiplier for CartPole and Acrobot lengths for learner environment")
parser.add_argument('--mass-mulE', type=float, default=1.0, metavar='G',
                    help="Multiplier for CartPole and Acrobot masses for expert environment")
parser.add_argument('--len-mulE', type=float, default=1.0, metavar='G',
                    help="Multiplier for CartPole and Acrobot lengths for expert environment")
args = parser.parse_args()

to_plot = []
to_plot_std = []
if "gail" in args.alg and not args.alg[0] == "gail":
    raise ValueError("gail must be passed as first argument in --alg")

if args.env_name == "gridworld-v0":
    subfolder = "env"+str(args.grid_type)+"noiseE"+str(args.noiseE)+"noiseL"+str(args.noiseL)
    print(subfolder)
    if not os.path.isdir(assets_dir(subfolder)):
        raise ValueError("No data for this environment")
    if not os.path.isdir("../plot/" + subfolder):
        os.makedirs("../plot/" + subfolder)
elif args.env_name == "HalfCheetahMuJoCoEnv-v0" or args.env_name == "HalfCheetahPyBulletEnv-v0":
    subfolder = "env" + args.env_name
    if not os.path.isdir(assets_dir(subfolder)):
        raise ValueError("No data for this environment")
    if not os.path.isdir("../plot/" + subfolder):
        os.makedirs("../plot/" + subfolder)
elif args.env_name == "Acrobot-v1":
    subfolder = "env" + args.env_name + "massL" + str(args.mass_mulL) + "massE" + str(args.mass_mulE) \
                    + "lenL" + str(args.len_mulL) + "lenE" + str(args.len_mulE)
    if not os.path.isdir(assets_dir(subfolder)):
        raise ValueError("No data for this environment")
    if not os.path.isdir("../plot/" + subfolder):
        os.makedirs("../plot/" + subfolder)
elif args.env_name == "HalfCheetah-v2" or args.env_name == "Ant-v2" or args.env_name == "Walker2d-v2":
    if not args.friction:
        subfolder = "env" + args.env_name + "massL" + str(args.mass_mulL) + "massE" + str(args.mass_mulE)
    else:
        subfolder = "env" + args.env_name + "frictionL" + str(args.mass_mulL) + "frictionE" + str(args.mass_mulE)
    if not os.path.isdir(assets_dir(subfolder)):
        raise ValueError("No data for this environment")
    if not os.path.isdir("../plot/" + subfolder):
        os.makedirs("../plot/" + subfolder)
elif args.env_name == "MountainCarContinuous-v0":
    subfolder = "env" + args.env_name + "powerL" + str(args.mass_mulL) + "powerE1.0"
    if not os.path.isdir(assets_dir(subfolder)):
        raise ValueError("No data for this environment")
    if not os.path.isdir("../plot/" + subfolder):
        os.makedirs("../plot/" + subfolder)
else:
    #subfolder = None
    subfolder = "env" + args.env_name
    if not os.path.isdir(assets_dir(subfolder)):
        raise ValueError("No data for this environment")
    if not os.path.isdir("../plot/" + subfolder):
        os.makedirs("../plot/" + subfolder)
for alg in args.alg:
    for alpha in args.alpha:
        list_tmp = []
        for seed in args.seeds:
            if  os.path.getsize(os.path.join(assets_dir(subfolder),
                                                   'reward_history/{}_{}_{}.p'.format(
                                                   args.env_name + str(seed), alg, alpha))) > 0:
                loaded = np.array(pickle.load(open(os.path.join(assets_dir(subfolder),
                                                   'reward_history/{}_{}_{}.p'.format(
                                                   args.env_name + str(seed),alg, alpha)), "rb")))

                list_tmp.append(loaded)


        print(np.array(list_tmp).shape)
        to_append = np.mean(np.array(list_tmp),axis=0)
        if len(args.seeds) > 1:
            to_append_std = np.std(np.array(list_tmp),axis=0)
        else:
            to_append_std = 0

        to_plot.append(to_append)
        to_plot_std.append(to_append_std)

if subfolder is not None:
    folder_plot = "../plot/"+subfolder+"/"
else:
    folder_plot = "../plot/"
plot.plot_lines_and_ranges(list_to_plot=to_plot,
                          list_sigmas=to_plot_std,
                          list_name= args.alpha,
                          axis_label=["IRL steps", "Total Reward"],
                          folder=folder_plot ,
                          title=alg[0]+"learning_curve_env" + args.env_name+str(args.seeds)+"friction"+str(args.friction))
