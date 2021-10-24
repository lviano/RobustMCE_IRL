import math
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
import copy
import random
from scipy import sparse

class TabularEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }


    def __init__(self, prop):

        self.viewer = None
        self.n_states = self.size**2
        self.n_actions = 4

        self.reward_range = (-100, 0)
        self.action_space = spaces.Discrete(4)
        # although there are 2 terminal squares in the grid
        # they are considered as 1 state
        # therefore observation is between 0 and 14
        self.observation_space = spaces.Box(low=0, high=20, shape=(self.size**2,), dtype=np.float32)




        self.features = np.eye(self.size**2)
        _, self.features_dim = self.features.shape

        # Initial state distribution
        self.p_in = np.ones(self.n_states) / self.n_states

        # Discount factor
        self.gamma = 0.99

        # Stochastic transitions
        self.prop_random_actions = prop



    def get_features(self, state=None, state_id=None):
        if state_id is None:
            state_id = self.state_to_index(state)
        return self.features[state_id]

    def compute_reward(self):
        self.state_r = self.features.dot(self.w)
        state_r = np.copy(self.state_r)
        self.compute_action_reward(state_r)

    def compute_action_reward(self, state_r):
        for i_state in range(self.n_states):
            pos_actions = self.get_possible_actions(state_id=i_state)
            for i_action in range(self.n_actions):
                if i_action in pos_actions:
                    self.R[i_action][i_state] = state_r[i_state]
                else:
                    self.R[i_action][i_state] = -np.inf

    def compute_transition_probs(self):
        self.P = np.zeros((self.n_actions, self.n_states, self.n_states))

        for i_state in range(self.n_states):
            poss_actions = self.get_possible_actions(state_id=i_state)

            poss_n_states = []
            for i_action in range(self.n_actions):
                if i_action in poss_actions:
                    i_n_state = self.state_to_index(self.take_action(self.index_to_state(i_state), i_action))
                    self.P[i_action][i_state][i_n_state] = 1 - self.prop_random_actions
                    poss_n_states.append(i_n_state)
                else:
                    self.P[i_action][i_state][i_state] += 1 - self.prop_random_actions
                    poss_n_states.append(i_state)

            # Random transitions
            for i_action in range(self.n_actions):
                for poss in poss_n_states:
                    self.P[i_action][i_state][poss] += self.prop_random_actions / len(poss_n_states)

        # Terminal states
        for i_action in range(self.n_actions):
            for i_state in self.terminal_indexes:
                self.P[i_action][i_state] = 0
                self.P[i_action][i_state][i_state] = 1

        # Convert to sparse matrix
        self.sparseP = {}
        for i_action in range(self.n_actions):
            self.sparseP[i_action] = sparse.csr_matrix(self.P[i_action])

    def get_random_initial_state(self):
        return self.index_to_state(np.random.choice(range(self.n_states), p=self.p_in))

    def random_policy(self):
        return np.array([random.choice(self.get_possible_actions(state_id=i_s)) for i_s in range(self.n_states)])

    def uniform_policy(self):
        return np.array([[1. / len(self.get_possible_actions(state_id=i_s)) if a in self.get_possible_actions(
            state_id=i_s) else 0 for a in range(self.n_actions)] for i_s in range(self.n_states)])

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


class GridWorldEnvironment(TabularEnv):
    def __init__(self, env_type=0, prop=0):
        # Characteristics of the gridworld
        self.size = 5

        self.actions = {
            0: np.array([-1, 0]),  # Up
            1: np.array([0, 1]),  # Right
            2: np.array([1, 0]),  # Down
            3: np.array([0, -1])  # Left
        }
        self.symb_actions = {
            0: "↑",
            1: "→",
            2: "↓",
            3: "←"
        }
        TabularEnv.__init__(self, prop)
        self.env_type = env_type
        self.seed()
        self.reset()
        # Reward
        self.w = self.generate_w_terminal(env_type)
        self.terminal_indexes = np.where(self.w == 0)[0]
        self.R = np.zeros((self.n_actions, self.n_states))
        self.compute_reward()

        # Transition probabilities
        self.compute_transition_probs()


    def get_transition_matrix(self):
        return self.P
    def get_starting_index(self):
        if self.env_type == 0:
            return 24
        if self.env_type == 1:
            return 24
        if self.env_type == 2:
            return 24
        if self.env_type == 9:
            return 24
        if self.env_type == 10:
            return 24
    def generate_w_terminal(self, env_type):
        w = -np.ones(self.features_dim)

        if env_type == 0:
            w[0] = 0
        elif env_type == 1:
            w_tmp = w.reshape(self.size, self.size)
            w_tmp[1:-1, 1:-1] = -100
            w = w_tmp.reshape(self.features_dim)
            w[0] = 0
        elif env_type == 2:
            w_tmp = w.reshape(self.size, self.size)
            w_tmp[int(0.5*(self.size-1)):int(0.5*self.size)+1, 1:-1] = -100
            w_tmp[1:-1, int(0.5*(self.size-1)):int(0.5*self.size)+1] = -100
            w = w_tmp.reshape(self.features_dim)
            w[0] = 0
        elif env_type == 3:
            w_tmp = w.reshape(self.size, self.size)
            w_tmp[int(0.5*(self.size-1)):int(0.5*self.size)+1, 1:-1] = -100
            w_tmp[1:-1, int(0.5*(self.size-1)):int(0.5*self.size)+1] = -100
            w_tmp[0, 4:6] = -50
            w = w_tmp.reshape(self.features_dim)
            w[0] = 0
        elif env_type == 4:
            w_tmp = w.reshape(self.size, self.size)
            w_tmp[int(0.5*(self.size-1)):int(0.5*self.size)+1, 1:-1] = -100
            w_tmp[1:-1, int(0.5*(self.size-1)):int(0.5*self.size)+1] = -100
            w_tmp[4:6, 0] = -50
            w = w_tmp.reshape(self.features_dim)
            w[0] = 0
        elif env_type == 5:
            w_tmp = w.reshape(self.size, self.size)
            w_tmp[[0,2,-3,-1], 2:-2] = -100
            w = w_tmp.reshape(self.features_dim)
            w[0] = 0
        elif env_type == 6:
            w_tmp = w.reshape(self.size, self.size)
            w_tmp[[0,2,-1], 1:-1] = -100
            w = w_tmp.reshape(self.features_dim)
            w[0] = 0
        elif env_type == 7:
            w_tmp = w.reshape(self.size, self.size)
            w_tmp[int(0.5*(self.size-1)):int(0.5*self.size)+1, int(0.5*(self.size-1)):int(0.5*self.size)+1] = -100
            w = w_tmp.reshape(self.features_dim)
            w[0] = 0
        elif env_type == 8:
            w_tmp = w.reshape(self.size, self.size)
            w_tmp[2:4, 2:4] = -100
            w = w_tmp.reshape(self.features_dim)
            w[0] = 0
        elif env_type == 9:
            w_tmp = w.reshape(self.size, self.size)
            w_tmp[[0,2,-3,-1], 2:-2] = -100
            w = w_tmp.reshape(self.features_dim)
            w[0] = 0

        elif env_type == 10:
            w_tmp = w.reshape(self.size, self.size)
            w_tmp[[0,2,4], 1:-1] = -100
            w = w_tmp.reshape(self.features_dim)
            w[0] = 0
        return w

    def is_in_grid(self, state=None, state_id=None):
        if state is None:
            state = self.index_to_state(state_id)
        return (state[0] >= 0) & (state[1] <= self.size - 1) & (state[0] <= self.size - 1) & (state[1] >= 0)

    def get_possible_actions(self, state=None, state_id=None):
        if state is None:
            state = self.index_to_state(state_id)
        av_actions = []
        for a in range(self.n_actions):
            if self.is_in_grid(state + self.actions[a]):
               av_actions.append(a)
        return av_actions

    def take_action(self, state, action):
        n_state = state + self.actions[action]
        if self.is_in_grid(n_state):
            return n_state
        else:
            return state

    def step(self, a):
        p_next_index = self.P[a][self.current_index]
        self.current_index = np.random.choice(np.arange(self.n_states), p=p_next_index)
        self.current_state = self.index_to_state(self.current_index)

        if self.current_index in self.terminal_indexes:
            self.done = True
        reward = self.state_r[self.current_index]

        return self.get_features(state_id = self.current_index), reward, self.done, None

    def reset(self, starting_index = None):
        starting_index = self.get_starting_index()
        if starting_index is None:
            self.current_state = self.get_random_initial_state()
            self.current_index = self.state_to_index( self.current_state)
        else:
            self.current_index = starting_index
            self.current_state = self.index_to_state(self.current_index)
        self.done = False
        return self.get_features(state_id = self.current_index)


    def state_to_index(self, state):
        return self.size*state[0] + state[1]

    def index_to_state(self, index):
        return np.array([int(index/self.size), index - self.size * int(index/self.size)])

    def get_full_rewards(self):
        state_r = self.features.dot(self.w)
        return(np.round(state_r.reshape(self.size, self.size), 2))

    """def display_policy_terminal(self, policy):
        pol = np.array([self.symb_actions[i] for i in policy])
        for s in self.terminal_indexes:
            pol[s] = "T"
        print(pol.reshape(self.size, self.size))

    def compute_reward_update(self, state_reward):
        for i_state in range(self.n_states):
            pos_actions = self.get_possible_actions(state_id=i_state)
            for i_action in range(self.n_actions):
                if i_action in pos_actions:
                    self.r[i_state][i_action] = state_reward[i_state]
                else:
                    self.r[i_state][i_action] = -np.inf
        return self.r

    def get_rewards(self, state, state_id=None):
        state_r = self.features.dot(self.w)
        if state_id is None:
            state_id = self.state_to_index(state)
        return state_r[state_id]"""