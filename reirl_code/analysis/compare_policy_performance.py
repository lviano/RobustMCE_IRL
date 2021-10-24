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

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', metavar='G',
                    help='name of the environment to run')
parser.add_argument("--alpha", nargs='+', default=["0"])
parser.add_argument("--seeds", nargs='+', default=["0"])
parser.add_argument("--noiseE", nargs='+', default=["0"],
                    help='expert noise')
parser.add_argument("--mass-muls", nargs='+', default=["0"],
                    help='expert masses')
parser.add_argument("--len-muls", nargs='+', default=["0"],
                    help='expert lengths')
parser.add_argument("--noiseL", type=float, metavar='G',
                    help='learner noise')
parser.add_argument("--grid-type", type=int, metavar='G',
                    help='learner noise')
parser.add_argument("--render", default=False, action='store_true')
parser.add_argument("--no-compute", default=False, action='store_true')
parser.add_argument("--var-mass", default=False, action='store_true')
parser.add_argument("--var-len", default=False, action='store_true')
parser.add_argument("--vi-expert", default=False, action='store_true')
parser.add_argument("--best", default=False, action='store_true')
parser.add_argument("--max-steps", type=int, default=5000)
args = parser.parse_args()


def evaluate_loop(policy_net, running_state, expert_flag):
    num_steps = 0
    episodes_reward = []

    for i_episode in count():

        state = env.reset()
        state_expert = copy.deepcopy(state)
        """if args.env_name == "gridworld-v0":
            s_index_expert = env.state_to_index(state_expert)"""
        state = running_state(state)
        reward_episode = 0

        for t in range(10000):
            state_var = tensor(state).unsqueeze(0).to(dtype)
            # choose mean action
            if expert_flag and args.env_name == "gridworld-v0":
                """if args.env_name == "gridworld-v0":
                    action = policy_net.select_action(s_index_expert)
                else:"""
                action = policy_net.select_action(state_expert)
            else:
                if not is_disc_action:
                    action = policy_net(state_var)[0][0].detach().numpy()
                else:
                    action = policy_net.select_action(state_var)[0].numpy()
            # action = policy_net.select_action(state_var)[0].cpu().numpy()
            action = int(action) if is_disc_action else action.astype(
                np.float64)
            next_state, reward, done, _ = env.step(action)
            state_expert = copy.deepcopy(next_state)
            """if args.env_name == "gridworld-v0":
                s_index_expert = env.state_to_index(state_expert)"""
            next_state = running_state(next_state)
            reward_episode += reward
            num_steps += 1

            if args.render:
                env.render()
            if done or num_steps >= args.max_steps:
                break

            state = next_state
        if done:
            episodes_reward.append(reward_episode)
            print('Episode {}\t reward: {:.2f}'.format(i_episode, reward_episode))

        if num_steps >= args.max_steps:
            break

    return np.mean(episodes_reward)


dtype = torch.float64
torch.set_default_dtype(dtype)
if not args.no_compute:
    to_plot = []
    to_plot_std = []

    if args.env_name == "GaussianGridworld-v0":
        env = gym.make(args.env_name, prop=args.noiseL, env_type=args.grid_type)
        subfolder = "env" + str(args.grid_type) + "noiseL" + str(args.noiseL)
        if not os.path.isdir(assets_dir(subfolder)):
            os.makedirs(assets_dir(subfolder))

        env.seed(0)
        torch.manual_seed(0)

        is_disc_action = len(env.action_space.shape) == 0
        state_dim = env.observation_space.shape[0]

        if not os.path.isdir("../plot/" + subfolder):
            os.makedirs("../plot/" + subfolder)

        for alpha in args.alpha:
            to_append = []
            to_append_std = []
            for noiseE in args.noiseE:
                means_avg = []
                stds_avg = []
                running_state = lambda x: x
                for seed in args.seeds:
                    print("First seed")
                    means = []
                    data_subfolder = "env" + args.env_name \
                                     + "type" + str(args.grid_type) \
                                     + "noiseE" + str(noiseE) \
                                     + "noiseL" + str(args.noiseL)
                    if not args.best:
                        policy_net = pickle.load(
                            open(os.path.join(assets_dir(data_subfolder),
                                              'learned_models/{}_REIRL_{}.p'.format(
                                                  args.env_name + str(seed),
                                                  alpha)),
                                 "rb"))
                    else:
                        policy_net = pickle.load(
                            open(os.path.join(assets_dir(data_subfolder),
                                              'learned_models/{}_REIRL_best_{}.p'.format(
                                                  args.env_name + str(seed),
                                                  alpha)),
                                 "rb"))
                        expert_flag = False

                    for _ in range(2):
                        mean_reward = evaluate_loop(policy_net,
                                                    running_state, False)
                        means.append(mean_reward)

                    means_avg.append(np.mean(means))
                    stds_avg.append(np.std(means))

                to_append.append(np.mean(means_avg))
                to_append_std.append(np.mean(stds_avg))
                #to_append_std.append(np.std(means_avg))

            to_append = np.array(to_append)
            to_append_std = np.array(to_append_std)
            to_plot.append(to_append)
            to_plot_std.append(to_append_std)

        names = args.alpha
        if not args.best:
            pickle.dump((to_plot, to_plot_std, names),
                        open("../plot/" + subfolder + "/DataCompareAlphas"
                             + args.env_name + str(
                            args.grid_type) + "noiseL" + str(
                            args.noiseL) + str(args.seeds) + ".p", 'wb'))
        else:
            pickle.dump((to_plot, to_plot_std, names),
                        open("../plot/" + subfolder + "/DataCompareAlphas"
                             + args.env_name + str(
                            args.grid_type) + "noiseL" + str(
                            args.noiseL) + str(args.seeds) + "best.p", 'wb'))

        plot.plot_lines_and_ranges(list_to_plot=to_plot,
                                   list_sigmas=to_plot_std,
                                   list_name=names,
                                   axis_label=["Noise E", "Total Reward"],
                                   folder="../plot/" + subfolder + "/",
                                   title="CompareAlphas" + args.env_name + str(
                                       args.grid_type) + "noiseL" + str(
                                       args.noiseL) + "best" + str(args.best) + str(
                                       args.seeds),
                                   x_axis=args.noiseE)
