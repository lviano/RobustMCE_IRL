#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import random
from scipy import sparse

import math
from itertools import product

class Environment:
    def __init__(self, n_states, n_actions, features, prop):
        # Characteristics of the environment
        self.n_states = n_states
        self.n_actions = n_actions
        self.features = features
        self.svf_features = np.eye(n_states)
        _, self.features_dim = features.shape

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

    def get_svf_features(self, state=None, state_id=None):
        if state_id is None:
            state_id = self.state_to_index(state)
        return self.svf_features[state_id]

    def compute_reward(self):
        self.state_r = self.features.dot(self.w)
        state_r = np.copy(self.state_r)
        self.compute_action_reward(state_r)

    def compute_action_reward(self, state_r):
        for i_state in range(self.n_states):
            pos_actions = self.get_possible_actions(state_id=i_state)
            for i_action in range(self.n_actions):
                if i_action in pos_actions:
                    self.r[i_state][i_action] = state_r[i_state]
                else:
                    self.r[i_state][i_action] = -np.inf


    def compute_transition_probs(self):
        self.T = np.zeros((self.n_actions, self.n_states, self.n_states))
        print(self.n_states)
        for i_state in range(self.n_states):
            poss_actions = self.get_possible_actions(state_id=i_state)
            poss_n_states = []
            for i_action in range(self.n_actions):
                if i_action in poss_actions:
                    i_n_state = self.state_to_index(self.take_action(self.index_to_state(i_state), i_action))
                    self.T[i_action][i_state][i_n_state] = 1 - self.prop_random_actions
                    poss_n_states.append(i_n_state)
                else:
                    self.T[i_action][i_state][i_state] = 1 - self.prop_random_actions
                    poss_n_states.append(i_state)
            # Random transitions
            for i_action in range(self.n_actions):
                self.T[i_action][i_state][poss_n_states] += self.prop_random_actions/len(poss_n_states)

        # Terminal states
        for i_action in range(self.n_actions):
            for i_state in self.terminal_indexes:
                self.T[i_action][i_state] = 0
                self.T[i_action][i_state][i_state] = 1

        # Convert to sparse matrix
        self.sparseT = {}
        for i_action in range(self.n_actions):
            self.sparseT[i_action] = sparse.csr_matrix(self.T[i_action])

    def get_random_initial_state(self):
        return self.index_to_state(np.random.choice(range(self.n_states), p=self.p_in))

    def random_policy(self):
        return np.array([random.choice(self.get_possible_actions(state_id=i_s)) for i_s in range(self.n_states)])

    def uniform_policy(self):
        return np.array([[1./len(self.get_possible_actions(state_id=i_s))  if a in self.get_possible_actions(state_id=i_s) else 0 for a in range(self.n_actions)] for i_s in range(self.n_states)])

class GridWorldEnvironment(Environment):
    def __init__(self, env_type, size, prop=0):
        # Characteristics of the gridworld
        self.size = size
        n_states = size**2

        n_actions = 4
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

        features = np.eye(n_states)

        Environment.__init__(self, n_states, n_actions, features, prop)

        # Reward
        self.w = self.generate_w_terminal(env_type)
        self.terminal_indexes = np.where(self.w == 0)[0]

        self.r = np.zeros((self.n_states, self.n_actions))
        self.compute_reward()

        # Transition probabilities
        self.compute_transition_probs()

    def get_transition_matrix(self):
        return self.T

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
            w_tmp[[0,2], 1:-1] = -100
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

    def state_to_index(self, state):
        return self.size*state[0] + state[1]

    def index_to_state(self, index):
        return np.array([int(index/self.size), index - self.size * int(index/self.size)])

    def display_rewards(self):
        state_r = self.features.dot(self.w)
        print(np.round(state_r.reshape(self.size, self.size), 2))

    def get_full_rewards(self):
        state_r = self.features.dot(self.w)
        return(np.round(state_r.reshape(self.size, self.size), 2))

    def display_policy_terminal(self, policy):
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
        return state_r[state_id]

class OWObject(object):
    """
    Object in objectworld.
    """

    def __init__(self, inner_colour, outer_colour):
        """
        inner_colour: Inner colour of object. int.
        outer_colour: Outer colour of object. int.
        -> OWObject
        """

        self.inner_colour = inner_colour
        self.outer_colour = outer_colour

    def __str__(self):
        """
        A string representation of this object.
        -> __str__
        """

        return "<OWObject (In: {}) (Out: {})>".format(self.inner_colour, self.outer_colour)


class ObjectWorldEnvironment(Environment):
    def __init__(self, size, n_objects, n_colours, seed, prop=0):
        random.seed(seed)
        # Characteristics of the gridworld
        self.size = size
        self.n_states = size**2
        self.n_actions = 5
        self.actions = {
            0: np.array([-1, 0]),  # Up
            1: np.array([0, 1]),  # Right
            2: np.array([1, 0]),  # Down
            3: np.array([0, -1]),  # Left
            4: np.array([0, 0])   #Stay
        }
        self.symb_actions = {
            0: "↑",
            1: "→",
            2: "↓",
            3: "←",
            4: "s"
        }
        self.n_objects = n_objects
        self.n_colours = n_colours

        # Generate objects.

        self.objects = {}
        for _ in range(self.n_objects):
            obj = OWObject(random.randint(0, self.n_colours-1),
                           random.randint(0, self.n_colours-1))

            while True:
                x = random.randint(0, self.size-1)
                y = random.randint(0, self.size-1)

                if (x, y) not in self.objects:
                    break

            self.objects[x, y] = obj

        features = self.feature_matrix()
        Environment.__init__(self, self.n_states, self.n_actions, features, prop)

        # Reward
        self.terminal_indexes = []
        self.r = np.zeros((self.n_states, self.n_actions))
        self.compute_reward()
        self.construct_rewards()

        # Transition probabilities
        self.compute_transition_probs()

    def feature_vector(self, i, discrete=True):
        """
        Get the feature vector associated with a state integer.
        i: State int.
        discrete: Whether the feature vectors should be discrete (default True).
            bool.
        -> Feature vector.
        """

        state = self.index_to_state(i)
        sx = state[0]
        sy = state[1]
        nearest_inner = {}  # colour: distance
        nearest_outer = {}  # colour: distance

        for y in range(self.size):
            for x in range(self.size):
                if (x, y) in self.objects:
                    dist = np.abs(x - sx) + np.abs(y - sy)
                    obj = self.objects[x, y]
                    if obj.inner_colour in nearest_inner:
                        if dist < nearest_inner[obj.inner_colour]:
                            nearest_inner[obj.inner_colour] = dist
                    else:
                        nearest_inner[obj.inner_colour] = dist
                    if obj.outer_colour in nearest_outer:
                        if dist < nearest_outer[obj.outer_colour]:
                            nearest_outer[obj.outer_colour] = dist
                    else:
                        nearest_outer[obj.outer_colour] = dist

        # Need to ensure that all colours are represented.
        for c in range(self.n_colours):
            if c not in nearest_inner:
                nearest_inner[c] = 0
            if c not in nearest_outer:
                nearest_outer[c] = 0

        if discrete:
            state = np.zeros((2*self.n_colours*self.size,))
            i = 0
            for c in range(self.n_colours):
                for d in range(1, self.size+1):
                    if nearest_inner[c] < d:
                        state[i] = 1
                    i += 1
                    if nearest_outer[c] < d:
                        state[i] = 1
                    i += 1
            assert i == 2*self.n_colours*self.size
            assert (state >= 0).all()
        else:
            # Continuous features.
            state = np.zeros((2*self.n_colours))
            i = 0
            for c in range(self.n_colours):
                state[i] = nearest_inner[c]
                i += 1
                state[i] = nearest_outer[c]
                i += 1

        return state

    def feature_matrix(self, discrete=True):
        """
        Get the feature matrix for this objectworld.
        discrete: Whether the feature vectors should be discrete (default True).
            bool.
        -> NumPy array with shape (n_states, n_states).
        """

        return np.array([self.feature_vector(i, discrete)
                         for i in range(self.n_states)])

    def reward_for_a_state(self, state_int):
        """
        Get the reward for a state int.
        state_int: State int.
        -> reward float
        """

        state = self.index_to_state(state_int)
        x = state[0]
        y = state[1]

        near_c0 = False
        near_c1 = False
        for (dx, dy) in product(range(-3, 4), range(-3, 4)):
            if 0 <= x + dx < self.size and 0 <= y + dy < self.size:

                if ((abs(dx) <= 2 and abs(dy) <= 2) and
                        (x+dx, y+dy) in self.objects and
                        self.objects[x+dx, y+dy].outer_colour == 0):
                    near_c0 = True
                if ((abs(dx) <= 1 and abs(dy) <= 1) and
                        (x+dx, y+dy) in self.objects and
                        self.objects[x+dx, y+dy].outer_colour == 1):
                    near_c1 = True

        if near_c0 and near_c1:
            return 1
        if near_c0:
            return -1
        return 0

    def get_transition_matrix(self):
        return self.T

    def construct_rewards(self):
        self.state_r = np.zeros((self.n_states))
        for i_state in range(self.n_states):
            self.state_r[i_state] = self.reward_for_a_state(i_state)
        return self.state_r

    def compute_reward(self):
        for i_state in range(self.n_states):
            pos_actions = self.get_possible_actions(state_id=i_state)
            for i_action in range(self.n_actions):
                if i_action in pos_actions:
                    self.r[i_state][i_action] = self.reward_for_a_state(i_state)
                else:
                    self.r[i_state][i_action] = -np.inf

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

    def state_to_index(self, state):
        return self.size*state[0] + state[1]

    def index_to_state(self, index):
        return np.array([int(index/self.size), index - self.size * int(index/self.size)])

    def display_rewards(self):
        r = self.state_r.copy()
        print(r.reshape((self.size, self.size)))

    def get_full_rewards(self):
        r = self.state_r.copy()
        return r.reshape((self.size, self.size))

    def display_policy_terminal(self, policy):
        pol = np.array([self.symb_actions[i] for i in policy])
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
        if state_id is None:
            state_id = self.state_to_index(state)
        return self.state_r[state_id]

class TwoDangerEnvironment(Environment):
    def __init__(self, env_type, size, prop=0):
        # Characteristics of the gridworld
        self.size = size
        n_states = size**2

        n_actions = 4
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

        features = self.generate_features(env_type, n_states)

        Environment.__init__(self, n_states, n_actions, features, prop)

        # Reward
        self.w = np.array([-1, -5, -1]) #  Cells with danger type 0 have reward -1
                             #  Cells with danger type 1 have reward -5
                             #  Cells that are not terminals receive additional - 1 reward
        self.terminal_indexes = np.where(self.features[:,2].flatten() == 0)[0]

        self.r = np.zeros((self.n_states, self.n_actions))
        self.compute_reward()

        # Transition probabilities
        self.compute_transition_probs()

    def get_transition_matrix(self):
        return self.T

    def generate_features(self, env_type, n_states):
        features = np.zeros([n_states, 3])

        if env_type == 0:
            features[:,[0,2]] = 1
            features[0,2] = 0
        elif env_type == 1:
            feature0 = np.zeros((self.size, self.size))
            feature1 = np.zeros((self.size, self.size))
            for i in np.arange(1,self.size-1):
                for j in np.arange(1,self.size-1):
                    if i > j:
                        feature0[i][j] = 1
                    else:
                        feature1[i][j] = 1

            features = np.hstack((feature0.reshape(n_states,1), feature1.reshape(n_states,1), np.ones([n_states,1])))
            features[0,2] = 0

        return features

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

    def state_to_index(self, state):
        return self.size*state[0] + state[1]

    def index_to_state(self, index):
        return np.array([int(index/self.size), index - self.size * int(index/self.size)])

    def display_rewards(self):
        state_r = self.features.dot(self.w)
        print(np.round(state_r.reshape(self.size, self.size), 2))

    def get_full_rewards(self):
        state_r = self.features.dot(self.w)
        return(np.round(state_r.reshape(self.size, self.size), 2))

    def display_policy_terminal(self, policy):
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
        return state_r[state_id]

class Inf_Horizon_ObjectWorldEnvironment(Environment):
    def __init__(self, size, n_objects, n_colours, seed, prop=0):
        random.seed(seed)
        # Characteristics of the gridworld
        self.size = size
        self.n_states = size**2
        self.gamma = 0.7
        self.n_actions = 4
        self.actions = {
            0: np.array([-1, 0]),  # Up
            1: np.array([0, 1]),  # Right
            2: np.array([1, 0]),  # Down
            3: np.array([0, -1])  # Left
#             4: np.array([0, 0])   #Stay
        }
        self.symb_actions = {
            0: "↑",
            1: "→",
            2: "↓",
            3: "←"
#             4: "s"
        }
        self.n_objects = n_objects
        self.n_colours = n_colours

        # Generate objects.

        self.objects = {}
        for _ in range(self.n_objects):
            obj = OWObject(random.randint(0, self.n_colours-1),
                           random.randint(0, self.n_colours-1))

            while True:
                x = random.randint(0, self.size-1)
                y = random.randint(0, self.size-1)

                if (x, y) not in self.objects:
                    break

            self.objects[x, y] = obj

        self.terminal_indexes = []
        # Reward
        self.r = np.zeros((self.n_states, self.n_actions))
        self.compute_reward()
        self.construct_rewards()
        self.get_terminal_index()

        features = self.feature_matrix()
        Environment.__init__(self, self.n_states, self.n_actions, features, prop)

        # Transition probabilities
        self.compute_transition_probs()

    def feature_vector(self, i, discrete=True):
        """
        Get the feature vector associated with a state integer.
        i: State int.
        discrete: Whether the feature vectors should be discrete (default True).
            bool.
        -> Feature vector.
        """

        state = self.index_to_state(i)
        sx = state[0]
        sy = state[1]
        nearest_inner = {}  # colour: distance
        nearest_outer = {}  # colour: distance

        for y in range(self.size):
            for x in range(self.size):
                if (x, y) in self.objects:
                    #dist = math.hypot((x - sx), (y - sy))
                    dist = np.abs(x - sx) + np.abs(y - sy)
                    obj = self.objects[x, y]
                    if obj.inner_colour in nearest_inner:
                        if dist < nearest_inner[obj.inner_colour]:
                            nearest_inner[obj.inner_colour] = dist
                    else:
                        nearest_inner[obj.inner_colour] = dist
                    if obj.outer_colour in nearest_outer:
                        if dist < nearest_outer[obj.outer_colour]:
                            nearest_outer[obj.outer_colour] = dist
                    else:
                        nearest_outer[obj.outer_colour] = dist

        # Need to ensure that all colours are represented.
        for c in range(self.n_colours):
            if c not in nearest_inner:
                nearest_inner[c] = 0
            if c not in nearest_outer:
                nearest_outer[c] = 0

        if discrete:
            state = np.zeros((2*self.n_colours*self.size,))
            i = 0
            for c in range(self.n_colours):
                for d in range(1, self.size + 1):
                    if nearest_inner[c] < d:
                        state[i] = 1
                    i += 1
                    if nearest_outer[c] < d:
                        state[i] = 1
                    i += 1
            assert i == 2*self.n_colours*self.size
            assert (state >= 0).all()
        else:
            # Continuous features.
            state = np.zeros((2*self.n_colours))
            i = 0
            for c in range(self.n_colours):
                state[i] = nearest_inner[c]
                i += 1
                state[i] = nearest_outer[c]
                i += 1

        return state

    def feature_matrix(self, discrete=True):
        """
        Get the feature matrix for this objectworld.
        discrete: Whether the feature vectors should be discrete (default True).
            bool.
        -> NumPy array with shape (n_states, n_states).
        """
        features = np.array([self.feature_vector(i, discrete) for i in range(self.n_states)])
        terminal_features = np.zeros((self.n_states, 1))
        terminal_features[self.terminal_indexes[0]] = 1
        return np.concatenate((features, terminal_features), axis = 1)
#         return features
    def reward_for_a_state(self, state_int):
        """
        Get the reward for a state int.
        state_int: State int.
        -> reward float
        """

        state = self.index_to_state(state_int)
        x = state[0]
        y = state[1]

        near_c0 = False
        near_c1 = False
        for (dx, dy) in product(range(-3, 4), range(-3, 4)):
            if 0 <= x + dx < self.size and 0 <= y + dy < self.size:
                """if (abs(dx) + abs(dy) <= 3 and
                        (x+dx, y+dy) in self.objects and
                        self.objects[x+dx, y+dy].outer_colour == 0):
                    near_c0 = True
                if (abs(dx) + abs(dy) <= 2 and
                        (x+dx, y+dy) in self.objects and
                        self.objects[x+dx, y+dy].outer_colour == 1):
                    near_c1 = True"""
                if ((abs(dx) <= 2 and abs(dy) <= 2) and
                        (x+dx, y+dy) in self.objects and
                        self.objects[x+dx, y+dy].outer_colour == 0):
                    near_c0 = True
                if ((abs(dx) <= 1 and abs(dy) <= 1) and
                        (x+dx, y+dy) in self.objects and
                        self.objects[x+dx, y+dy].outer_colour == 1):
                    near_c1 = True

        if near_c0 and near_c1:
            return 0
        if near_c0:
            return -2
        return -1

    def get_terminal_index(self):
        loc = np.where(self.state_r == 0)[0]
        if list(loc):
            self.terminal_indexes = random.sample(list(loc), 1)

    def get_transition_matrix(self):
        return self.T

    def construct_rewards(self):
        self.state_r = np.zeros((self.n_states))
        for i_state in range(self.n_states):
            self.state_r[i_state] = self.reward_for_a_state(i_state)
        return self.state_r

    def compute_reward(self):
        for i_state in range(self.n_states):
            pos_actions = self.get_possible_actions(state_id=i_state)
            for i_action in range(self.n_actions):
                if i_action in pos_actions:
                    self.r[i_state][i_action] = self.reward_for_a_state(i_state)
                else:
                    self.r[i_state][i_action] = -np.inf

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

    def state_to_index(self, state):
        return self.size*state[0] + state[1]

    def index_to_state(self, index):
        return np.array([int(index/self.size), index - self.size * int(index/self.size)])

    def display_rewards(self):
        r = self.state_r.copy()
        print(r.reshape((self.size, self.size)))

    def get_full_rewards(self):
        r = self.state_r.copy()
        return r.reshape((self.size, self.size))

    def display_policy_terminal(self, policy):
        pol = np.array([self.symb_actions[i] for i in policy])
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
        if state_id is None:
            state_id = self.state_to_index(state)
        return self.state_r[state_id]
