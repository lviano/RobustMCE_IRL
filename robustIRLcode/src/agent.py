#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
# from MDPsolver import MDPsolver
import random
import itertools


class Agent:
    def __init__(self, env, policy=None):
        self.env = env
        self.state = self.env.index_to_state(np.random.randint(self.env.n_states))
        self.policy = self.env.random_policy() if policy is None else policy

    def choose_action(self):
        # Random transitions
        if np.random.rand() < self.env.prop_random_actions:
            #return np.random.randint(self.env.n_actions)
            return np.random.choice(self.env.get_possible_actions(state = self.state))

        if len(self.policy.shape) > 1:
            # Stochastic policy
            prob = self.policy[self.env.state_to_index(self.state)]

            return np.random.choice(self.env.n_actions, p=prob)
        else:
            # Deterministic policy
            return self.policy[self.env.state_to_index(self.state)]

    def run_trajectory(self, limit=1e3, starting_state=None):
        """Run a trajectory and collect visited states, actions and features expectation"""

        self.state = self.env.get_random_initial_state() if starting_state is None else self.env.index_to_state(starting_state)

        traj_states = [np.copy(self.state)]
        traj_actions = []
        features_exp = np.zeros(self.env.features_dim)
        svf_exp = np.zeros(self.env.n_states)
        step = 0
        rewards = 0

        while True:
            if (step > limit) | (self.env.state_to_index(self.state) in self.env.terminal_indexes):
                break
            a = self.choose_action()
            traj_actions.append(a)
            features_exp += self.env.gamma**step * self.env.get_features(state=self.state)
            svf_exp += self.env.gamma**step * self.env.get_svf_features(state=self.state)
            rewards += self.env.gamma**step * self.env.get_rewards(state=self.state)
            self.state = self.env.take_action(self.state, a)
            traj_states.append(np.copy(self.state))

            step += 1

        # We complete the features expectation vector if terminal state is reached / trajectory too long
        svf_exp += ((self.env.gamma**step)/(1-self.env.gamma))*self.env.get_svf_features(state=self.state)
        features_exp += ((self.env.gamma**step)/(1-self.env.gamma))*self.env.get_features(state=self.state)

        return np.array(traj_states), np.array(traj_actions), (1-self.env.gamma)*features_exp, (1-self.env.gamma)*svf_exp, rewards

    def collect_trajectories(self, n_traj, limit, start = None):
        """Collect a batch of trajectories"""

        states, actions, features_exp, svf_exp, rewards = [], [], [], [], []
        """ Generate starting state"""

        for idx in range(n_traj):
            traj_states, traj_actions, mu_traj, svf_traj, traj_rewards = self.run_trajectory(limit=limit, starting_state=start)
            states.append(traj_states)
            actions.append(traj_actions)
            features_exp.append(mu_traj)
            svf_exp.append(svf_traj)
            rewards.append(traj_rewards)

        return np.array(states), np.array(actions), np.array(features_exp), np.array(svf_exp), np.array(rewards)

    def compute_trajectory_reward(self, limit = 1e3, starting_index = None):

        " Average the total return of the policy on n_traj trajectories"
        self.index = self.env.state_to_index(self.env.get_random_initial_state()) if starting_index is None else starting_index
        self.state = self.env.index_to_state(self.index)
        tot_reward = self.env.state_r[self.index]
        step = 1
        while True:
            if (step > limit) | (self.index in self.env.terminal_indexes):
                break
            a = self.choose_action()
            self.state = self.env.take_action(self.state, a)
            self.index = self.env.state_to_index(self.state)
            #tot_reward += self.env.reward[self.index] For ObjectWorld
            tot_reward += self.env.state_r[self.index]
            step += 1

        return tot_reward

    def evaluate_policy(self, n_trajectories, limit = 1e3, starting_index = None):
        returns = []
        for n in range(n_trajectories):
            if starting_index is None:
                returns.append(self.compute_trajectory_reward(limit = limit, starting_index = starting_index))
            else:
                returns.append(self.compute_trajectory_reward(limit = limit, starting_index = starting_index[n]))

        return np.mean(returns)
