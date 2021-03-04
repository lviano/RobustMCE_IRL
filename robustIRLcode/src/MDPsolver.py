#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
np.seterr(divide='ignore', invalid='ignore')
from scipy import sparse
from utils import softmax, softmax_probs
from agent import Agent
from scipy import special

class MDPsolver:
    def __init__(self, env):
        self.env = env

        self.v = np.zeros(self.env.n_states)                         # Value function
        self.policy = self.env.random_policy()                      # Policy
        self.oldQavailable = False

    def value_iteration(self, tol=1e-10):
        self.v = np.zeros(self.env.n_states)
        self.q = np.zeros((self.env.n_states, self.env.n_actions))

        while True:
            v_old = np.copy(self.v)
            for a in range(self.env.n_actions):
                self.q[:, a] = self.env.r[:, a] + self.env.gamma * self.env.sparseT[a].dot(self.v)
            self.v = np.max(self.q, axis=1)
            if np.linalg.norm(self.v - v_old) < tol:
                break
        self.policy = np.argmax(self.q, axis=1)

    def value_iteration_fixed_horizon(self, horizon = 10, tol=1e-10):
        self.v = np.zeros(self.env.n_states)
        self.q = np.zeros((self.env.n_states, self.env.n_actions))

        for _ in range(horizon):
            for a in range(self.env.n_actions):
                self.q[:, a] = self.env.r[:, a] + self.env.gamma * self.env.sparseT[a].dot(self.v)
            self.v = np.max(self.q, axis=1)
        self.policy = np.argmax(self.q, axis=1)

    def soft_value_iteration(self, tol=1e-10):
        self.v = np.zeros(self.env.n_states)
        self.q = np.zeros((self.env.n_states, self.env.n_actions))

        while True:
            v_old = np.copy(self.v)
            for a in range(self.env.n_actions):
                self.q[:, a] = self.env.r[:, a] + self.env.gamma * self.env.sparseT[a].dot(self.v)
            self.v = softmax(self.q).reshape(self.env.n_states)
            if np.linalg.norm(self.v - v_old) < tol:
                break
        return softmax_probs(self.q)

    def soft_value_iteration_fixed_horizon(self, horizon = 10, tol=1e-10):
        self.v = np.zeros(self.env.n_states)
        self.q = np.zeros((self.env.n_states, self.env.n_actions))

        for _ in range(horizon):
            for a in range(self.env.n_actions):
                self.q[:, a] = self.env.r[:, a] + self.env.gamma * self.env.sparseT[a].dot(self.v)
            self.v = softmax(self.q).reshape(self.env.n_states)

        return softmax_probs(self.q)

    def soft_2player_value_iteration(self, alpha, tol=1e-10):
        self.v = np.zeros(self.env.n_states)
        self.q2p = np.zeros((self.env.n_states, self.env.n_actions, self.env.n_actions))

        while True:
            v_old = np.copy(self.v)
            for a in range(self.env.n_actions):
                for o in range(self.env.n_actions):
                    self.q2p[:, a, o] = self.env.r[:, a] + self.env.gamma *((alpha)*self.env.sparseT[a].dot(self.v) + (1 - alpha)*self.env.sparseT[o].dot(self.v))
            self.v = softmax(np.min(self.q2p, axis = 2)).reshape(self.env.n_states)
            if np.linalg.norm(self.v - v_old) < tol:
                break
        opponent_policy = np.zeros((self.env.n_states, self.env.n_actions))
        q_opponent = special.logsumexp(self.q2p, axis = 1)
        for index in range(self.env.n_states):
            possible_actions = np.array(self.env.get_possible_actions(state = self.env.index_to_state(index)))
            worst_action_indices = np.where(q_opponent[index,possible_actions] == np.min(q_opponent[index,possible_actions]),True,False)
            worst_actions = possible_actions[np.array(worst_action_indices)]
            opponent_policy[index, :] = np.array([1/len(worst_actions) if a in worst_actions else 0 for a in range(self.env.n_actions)])

        return softmax_probs(np.min(self.q2p, axis = 2)), opponent_policy 

    def soft_2player_value_iteration_fixed_horizon(self, alpha, horizon = 10,  tol=1e-10):
        self.v = np.zeros(self.env.n_states)
        self.q2p = np.zeros((self.env.n_states, self.env.n_actions, self.env.n_actions))

        for _ in range(horizon):
            for a in range(self.env.n_actions):
                for o in range(self.env.n_actions):
                    self.q2p[:, a, o] = self.env.r[:, a] + self.env.gamma *((alpha)*self.env.sparseT[a].dot(self.v) + (1 - alpha)*self.env.sparseT[o].dot(self.v))
            self.v = softmax(np.min(self.q2p, axis = 2)).reshape(self.env.n_states)
            

        opponent_policy = np.zeros((self.env.n_states, self.env.n_actions))
        q_opponent = special.logsumexp(self.q2p, axis = 1)
        for index in range(self.env.n_states):
            possible_actions = np.array(self.env.get_possible_actions(state = self.env.index_to_state(index)))
            worst_action_indices = np.where(q_opponent[index,possible_actions] == np.min(q_opponent[index,possible_actions]),True,False)
            worst_actions = possible_actions[np.array(worst_action_indices)]
            opponent_policy[index, :] = np.array([1/len(worst_actions) if a in worst_actions else 0 for a in range(self.env.n_actions)])

        return softmax_probs(np.min(self.q2p, axis = 2)), opponent_policy

    def two_players_soft_Q(self, alpha, beta, beta_op, n_episodes, lr, tol=1e-10, reuseQ = False):

        if (not reuseQ) or not self.oldQavailable:
            self.q2p = np.zeros((self.env.n_states, self.env.n_actions, self.env.n_actions))
            self.q_player = np.zeros((self.env.n_states, self.env.n_actions))
            self.q_opponent = np.zeros((self.env.n_states, self.env.n_actions))
            for i in range(self.env.n_states):
                for a in range(self.env.n_actions):
                    for o in range(self.env.n_actions):
                        if not (a in self.env.get_possible_actions(state_id=i)):
                            self.q2p[i, a, :] = np.inf
                            self.q_player[i, a] = -np.inf
                            self.q_opponent[i, a] = np.inf
                            if not (o in self.env.get_possible_actions(state_id=i)):
                                self.q2p[i, :, o] = np.inf

        
        player = Agent(self.env, policy = self.env.uniform_policy()) #Stochastic Policy
        opponent = Agent(self.env, policy= self.env.uniform_policy())
        counter = 0
        for _ in range(n_episodes):
            step = 0
            common_state = player.env.get_random_initial_state()
            player.state = common_state
            opponent.state = common_state
            player.index = player.env.state_to_index(player.state)
            opponent.index = player.env.state_to_index(opponent.state)
            delta = 0
            while True:
                step += 1
                if (player.index in self.env.terminal_indexes or step > 1000):
                    if delta < tol and not delta == 0:
                        counter += 1
                        if counter == 30:
                            return player.policy, opponent.policy
                    break
            
                a = player.choose_action()
                player_next_state = player.env.take_action(player.state, a)
                player_next_is = player.env.state_to_index(player_next_state)

                o = opponent.choose_action()
                opponent_next_state = opponent.env.take_action(opponent.state, o)
                opponent_next_is = opponent.env.state_to_index(opponent_next_state)

                tot_reward = (1 - alpha) * opponent.env.r[opponent.index, o] + alpha * player.env.r[player.index, a]

                #TD Update
                update = (tot_reward + self.env.gamma*((1-alpha)*self.v[opponent_next_is] + alpha*self.v[player_next_is])-self.q2p[player.index, a, o])

                self.q2p[player.index, a, o] += lr * update

                delta = np.max([delta, np.abs(update)])

                possible_actions = np.array(opponent.env.get_possible_actions(state = opponent.state))
                #Marginalize Q values
                self.q_player[player.index, a] = -beta_op * softmax((self.q2p[player.index, a, possible_actions].reshape(1,-1))/-beta_op)
                self.q_opponent[opponent.index, o] = beta * softmax((self.q2p[opponent.index, possible_actions, o].reshape(1,-1))/beta)

                #Update V value
                self.v[player.index] = beta * softmax((self.q_player[player.index, possible_actions].reshape(1,-1))/beta)

                #Update Policies
                # """ Adversary """
                # opponent.policy[opponent.index, :] = softmax_probs(1/-beta_op*self.q_opponent[opponent.index, :].reshape(1,-1)) #/alpha)
                # """ Player """
                # player.policy[player.index, :] = softmax_probs(1/beta*self.q_player[player.index, :].reshape(1,-1)) #/alpha)

                poss_actions_prob = softmax_probs(1/beta*self.q_player[player.index, possible_actions].reshape(1,-1)) #/alpha)
                probs = np.zeros([player.env.n_actions])
                probs[possible_actions] = poss_actions_prob
                player.policy[player.index, :] = probs

                poss_actions_prob = softmax_probs(1/-beta_op*self.q_opponent[opponent.index, possible_actions].reshape(1,-1)) #/alpha)
                probs = np.zeros([player.env.n_actions])
                probs[possible_actions] = poss_actions_prob
                opponent.policy[opponent.index, :] = probs

                next_is = np.random.choice([player_next_is, opponent_next_is], p=[alpha, 1 - alpha])
                next_state = player.env.index_to_state(next_is)

                player.state = next_state
                opponent.state = next_state
                player.index = next_is
                opponent.index = next_is
        self.oldQavailable = True
        return player.policy, opponent.policy

    def rational_opponent_two_players_soft_Q(self, alpha, tol=1e-10, reuseQ = False):
        #Opponent takes control w.p. 1 - alpha
        self.lr = 0.5 #0.5 for deterministic environments, 0.1 for stochastic transitions
        if (not reuseQ) or not self.oldQavailable:
            self.q2p = np.zeros((self.env.n_states, self.env.n_actions, self.env.n_actions))
            self.q_player = np.zeros((self.env.n_states, self.env.n_actions))
            self.q_opponent = np.zeros((self.env.n_states, self.env.n_actions))
            for i in range(self.env.n_states):
                for a in range(self.env.n_actions):
                    for o in range(self.env.n_actions):
                        if not (a in self.env.get_possible_actions(state_id=i)):
                            self.q2p[i,a,:] = -np.inf
                            self.q_player[i,a] = -np.inf
                            self.q_opponent[i,a] = -np.inf
                            if not (o in self.env.get_possible_actions(state_id=i)):
                                self.q2p[i,:,o] = -np.inf
        else:
            print("Recycling Q's")
        n_episodes = 30000
        player = Agent(self.env, policy = self.env.uniform_policy()) #Stochastic Policy
        opponent = Agent(self.env, policy= self.env.uniform_policy())
        counter = 0
        for _ in range(n_episodes):
            common_state = player.env.get_random_initial_state()
            player.state = common_state
            opponent.state = common_state
            player.index = self.env.state_to_index(player.state)
            opponent.index = player.index
            delta = 0
            while True:
                
                if (player.index in self.env.terminal_indexes):

                    if delta < tol and not delta == 0:
                        counter += 1
                        if counter == 30:
                            """Debug Plots"""
                            self.oldQavailable = True
                            return player.policy, opponent.policy
                    break
                a = player.choose_action()
                player_next_state = player.env.take_action(player.state, a)
                player_next_is = player.env.state_to_index(player_next_state)

                o = opponent.choose_action()
                opponent_next_state = opponent.env.take_action(opponent.state, o)
                opponent_next_is = opponent.env.state_to_index(opponent_next_state)

                tot_reward = (1 - alpha)*opponent.env.r[opponent.index,o] + alpha*player.env.r[player.index,a]

                #TD Update
                update = (tot_reward + self.env.gamma*((1-alpha)*self.v[opponent_next_is] + alpha*self.v[player_next_is]) - self.q2p[player.index, a, o])
                
                
                self.q2p[player.index, a, o] += self.lr*update
                
                delta = np.max([delta, np.abs(update)])
                #Marginalize Q values
                possible_actions = np.array(opponent.env.get_possible_actions(state = opponent.state))
                self.q_player[player.index,a] = np.min(self.q2p[player.index,a,possible_actions])
                
                self.q_opponent[opponent.index,o] = softmax(self.q2p[opponent.index,possible_actions,o].reshape(1,-1)-np.log(self.env.n_actions))
                
                self.v[player.index] = softmax(self.q_player[player.index,possible_actions].reshape(1,-1)-np.log(self.env.n_actions))
                
                #Update Policies

                """ Deterministic Opponent here below """
                #opponent.policy[opponent.index] = possible_actions[np.argmin(self.q_opponent[opponent.index,possible_actions])]

                """ Tie breaking Opponent here below """
                #worst_action_indices = np.where(np.round(self.q_opponent[opponent.index,possible_actions],2) == np.min(np.round(self.q_opponent[opponent.index,possible_actions],2)),True,False)
                #worst_actions = possible_actions[np.array(worst_action_indices)]
                #opponent.policy[opponent.index, :] = np.array([1/len(worst_actions) if a in worst_actions else 0 for a in range(self.env.n_actions)])

                """Eps Greedy Opponent"""
                epsilon = 0.1
                worst_action_indices = np.where(np.round(self.q_opponent[opponent.index,possible_actions],2) == np.min(np.round(self.q_opponent[opponent.index,possible_actions],2)),True,False)
                worst_actions = possible_actions[np.array(worst_action_indices)]
                if len(possible_actions) == len(worst_actions):
                    opponent.policy[opponent.index, :] = np.array([1/len(worst_actions) if a in worst_actions else 0 for a in range(self.env.n_actions)])
                else:
                    opponent.policy[opponent.index, :] = np.array([(1 - epsilon)/len(worst_actions)
                                                            if a in worst_actions
                                                            else epsilon/(len(possible_actions) - len(worst_actions))
                                                            if (not(a in worst_actions) and a in possible_actions)
                                                            else 0 for a in range(self.env.n_actions)])
                """ Entropy Opponent """
                #poss_actions_prob = softmax_probs(-1*self.q_opponent[player.index,possible_actions].reshape(1,-1))
                #probs = np.zeros([self.env.n_actions])
                #probs[possible_actions] = poss_actions_prob
                #opponent.policy[opponent.index, :] = probs
                """ Entropy Player """
                poss_actions_prob = softmax_probs(self.q_player[player.index,possible_actions].reshape(1,-1)) #/alpha)
                probs = np.zeros([self.env.n_actions])
                probs[possible_actions] = poss_actions_prob
                player.policy[player.index] = probs

                next_is = np.random.choice([player_next_is, opponent_next_is], p=[alpha, 1 - alpha])

                next_state = player.env.index_to_state(next_is)

                player.state = next_state
                opponent.state = next_state  #both the agents act from the same state
                player.index = next_is
                opponent.index = next_is

        self.oldQavailable = True
        return player.policy, opponent.policy
    

    def mu_policy(self, policy, stochastic=False, emp_p_in=None, tol=1e-10, no_one_hot = False):

        
        if stochastic:
            p_policy = np.array([policy[s].dot(self.env.T[:, s, :]) for s in range(self.env.n_states)])
        else:
            p_policy = np.array([self.env.T[policy[s]][s] for s in range(self.env.n_states)])
        p_policy_sparse = sparse.csr_matrix(p_policy.T)

        sv = np.zeros(self.env.n_states)

        p_in = self.env.p_in if emp_p_in is None else emp_p_in

        while True:
            sv_old = np.copy(sv)
            sv = p_in + self.env.gamma * p_policy_sparse.dot(sv)
            if np.linalg.norm(sv - sv_old) < tol:
                break
        if not no_one_hot:
            return (1 - self.env.gamma) * sv.dot(self.env.svf_features)
        else:
            return (1 - self.env.gamma) * sv.dot(self.env.features)


    def mu_policy_fixed_horizon(self, policy, horizon = 10, stochastic=False, emp_p_in=None, tol=1e-10, no_one_hot = False):

        
        if stochastic:
            p_policy = np.array([policy[s].dot(self.env.T[:, s, :]) for s in range(self.env.n_states)])
        else:
            p_policy = np.array([self.env.T[policy[s]][s] for s in range(self.env.n_states)])
        p_policy_sparse = sparse.csr_matrix(p_policy.T)

        sv = np.ones(self.env.n_states)/self.env.n_states
        sum_sv = np.ones(self.env.n_states)/self.env.n_states
        p_in = self.env.p_in if emp_p_in is None else emp_p_in

        for _ in range(horizon):
            sv = self.env.gamma * p_policy_sparse.dot(sv)
            sum_sv += sv

        if not no_one_hot:
            return sum_sv.dot(self.env.svf_features)
        else:
            return sum_sv.dot(self.env.features)

    def mu_w(self, w, emp_p_in=None, two_players = False, alpha = 0.9, no_one_hot = False, fix_horizon = False):
        self.env.w = w
        self.env.compute_reward()
        if two_players:
            if fix_horizon:
                player_policy,opponent_policy = self.soft_2player_value_iteration(alpha)
                stochastic_pol = alpha*player_policy + (1-alpha)*opponent_policy
                return self.mu_policy_fixed_horizon(stochastic_pol, stochastic=True, emp_p_in=emp_p_in, no_one_hot=no_one_hot), player_policy, opponent_policy
            else:

                player_policy,opponent_policy = self.soft_2player_value_iteration(alpha)
                stochastic_pol = alpha*player_policy + (1-alpha)*opponent_policy
                return self.mu_policy(stochastic_pol, stochastic=True, emp_p_in=emp_p_in, no_one_hot=no_one_hot), player_policy, opponent_policy

        else:
            
            if fix_horizon:
                stochastic_pol = self.soft_value_iteration()
                return self.mu_policy_fixed_horizon(stochastic_pol, stochastic=True, emp_p_in=emp_p_in, no_one_hot=no_one_hot), stochastic_pol
            else:
                stochastic_pol = self.soft_value_iteration()
                return self.mu_policy(stochastic_pol, stochastic=True, emp_p_in=emp_p_in, no_one_hot=no_one_hot), stochastic_pol
