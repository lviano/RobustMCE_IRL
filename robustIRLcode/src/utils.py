import numpy as np
import pickle
import torch
import matplotlib.pyplot as plt

def softmax(x):
    return x.max(axis=1).reshape(x.shape[0], 1) + np.log(np.exp(x - x.max(axis=1).reshape(x.shape[0], 1)).sum(axis=1)).reshape(x.shape[0], 1)


def softmax_probs(x):
    return np.exp(x-np.max(x, axis=1).reshape(x.shape[0], 1)) / np.exp(x-np.max(x, axis=1).reshape(x.shape[0], 1)).sum(axis=1).reshape(x.shape[0], 1)


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     torch.backends.cudnn.deterministic = True


def get_mimic_policy(env, trajectories, actions):
    mimic_policy = np.zeros((env.n_states, env.n_actions))
    for traj in range(len(trajectories)):
        for step in range(len(actions[traj])):
            mimic_policy[env.state_to_index(trajectories[traj][step])][actions[traj][step]] += 1
    for idx in range(env.n_states):
        if np.sum(mimic_policy[idx]) == 0:
            mimic_policy[idx][env.get_possible_actions(state_id=idx)] = 1
    mimic_policy /= np.sum(mimic_policy, axis=1).reshape(env.n_states, 1)

    return mimic_policy


def save_result(result, name):
    with open('../results/'+ name + '.pkl', 'wb') as f:
        pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)

def load_result(name):
    with open('../results/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def randomize_optimal_policy(solver):
    randomized_pi = [[1 if solver.q[s,a] == np.max(solver.q[s,:]) else 0 for a in range(solver.env.n_actions)] for s in range(solver.env.n_states)]
    randomized_pi = randomized_pi/np.tile(np.sum(randomized_pi, axis = 1).reshape(solver.env.n_states,1) , solver.env.n_actions)
    return randomized_pi

def compute_initial_probabilities(demo_states, env):
    n_traj = len(demo_states)
    prob = np.zeros(env.n_states)
    init_states_list = np.array([env.state_to_index(demo_states[i][0]) for i in range(n_traj)])
    prob = [init_states_list[init_states_list == s].size for s in range(env.n_states)]
    prob = np.array(prob)*1/len(demo_states)
    return prob

def add_arrow(pi, shape, mode):
    if mode == "single":
        for s, a in enumerate(pi):    #acs optimal actions

            if a == 0: ##up
                plt.arrow(np.mod(s, shape[1]), int(s / shape[1]), 0, -0.45, head_width=0.05)
            if a == 1: ##right
                plt.arrow(np.mod(s, shape[1]), int(s / shape[1]),  0.45, 0, head_width=0.05)
            if a == 2: ##down
                plt.arrow(np.mod(s, shape[1]), int(s / shape[1]),  0, 0.45, head_width=0.05)
            if a == 3: ##left
                plt.arrow(np.mod(s, shape[1]), int(s / shape[1]),  -0.45, 0, head_width=0.05)
    if mode == "multiple":
        for s, acs in enumerate(pi):
            for a in acs:
                if a == 0: ##up
                    plt.arrow(np.mod(s, shape[1]), int(s / shape[1]), 0, -0.45, head_width=0.05)
                if a == 1: ##right
                    plt.arrow(np.mod(s, shape[1]), int(s / shape[1]),  0.45, 0, head_width=0.05)
                if a == 2: ##down
                    plt.arrow(np.mod(s, shape[1]), int(s / shape[1]),  0, 0.45, head_width=0.05)
                if a == 3: ##left
                    plt.arrow(np.mod(s, shape[1]), int(s / shape[1]),  -0.45, 0, head_width=0.05)

def load_data(base_dir, lr):
    with open(base_dir + "/policy_" + str(lr), "rb") as fp:
        policies = pickle.load(fp)
    with open(base_dir + "/player_" + str(lr), "rb") as fp:
        player = pickle.load(fp)
    with open(base_dir + "/adv_" + str(lr), "rb") as fp:
        adv = pickle.load(fp)
    with open(base_dir + "/reward_" + str(lr), "rb") as fp:
        reward = pickle.load(fp)

    data = {"policies":policies, "player":player, "adv":adv, "reward":reward}
    return data

def log_sum_exp(value, dim, keepdim):
    """Numerically stable implementation of the operation:
    value.exp().sum(dim, keepdim).log()
    """
    m, _ = torch.max(value, dim=dim, keepdim=True)
    value0 = value - m
    if keepdim is False:
        m = m.squeeze(dim)
    return m + torch.log(torch.sum(torch.exp(value0),
                                   dim=dim, keepdim=keepdim))

def load_data_iil(base_dir, lr):
    with open(base_dir + "/policy_" + str(lr), "rb") as fp:
        policies = pickle.load(fp)
    with open(base_dir + "/reward_" + str(lr), "rb") as fp:
        reward = pickle.load(fp)

    data = {"policies":policies, "reward":reward}
    return data
