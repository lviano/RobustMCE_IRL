#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from optimizers import GDOptimizer, AdamOptimizer
from RewardNet import RewardNet,adjust_learning_rate
from plot import plot_on_grid, plot_value_and_policy
from agent import Agent

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import pickle


class IRLsolver:
    def __init__(self, solver, mu_teacher=None, w_in=None):
        self.solver = solver
        self.mu_teacher = mu_teacher
        if w_in is None:
            w_in = np.zeros(self.solver.env.features_dim)
        self.w = w_in
    def max_ent_irl(self, base_dir, n_iter=200, tol=1e-10, verbose=True, optimizer="adam", lr=0.1, lr_order=1, emp_p_in = None, beta = 0, beta_op = 0, softQ_lr = 0.5, n_steps = 150, two_players=False, alpha = 0.9, reg_opp = False, no_one_hot = True, fix_horizon = False):
        if optimizer == "gd":
            opt = GDOptimizer(lr, lr_order)
        elif optimizer == "adam":
            opt = AdamOptimizer(self.solver.env.features_dim, lr)

        policies = []
        player_policies = []
        adv_policies = []
        rewards = []
        err = []
        

        if not reg_opp:
            mu_learner, max_ent_policy,  opponent_policy = self.solver.mu_w(self.w, emp_p_in, two_players, alpha, no_one_hot = no_one_hot, fix_horizon = fix_horizon)
            policy = alpha * max_ent_policy + (1-alpha) * opponent_policy
        else:
            max_ent_policy,  opponent_policy = self.solver.two_players_soft_Q(alpha = alpha, beta = beta, beta_op = beta_op, n_episodes = 1000, lr = softQ_lr, reuseQ=False)
            policy = alpha * max_ent_policy + (1-alpha) * opponent_policy
            if fix_horizon:
                mu_learner = self.solver.mu_policy_fixed_horizon(policy, stochastic=True, emp_p_in=emp_p_in, no_one_hot = no_one_hot)
            else:
                mu_learner = self.solver.mu_policy(policy, stochastic=True, emp_p_in=emp_p_in, no_one_hot = no_one_hot)
            
        rewards.append(np.copy(self.w))
        player_policies.append(max_ent_policy)
        adv_policies.append(opponent_policy)
        policies.append(policy)
        while opt.step < n_iter:
            # Update on w
            grad = self.mu_teacher - mu_learner
            self.w += opt.update(grad)
            print("Weights")
            print(self.w)
            # Update features expectation

            if not reg_opp:
                mu_learner, max_ent_policy,  opponent_policy = self.solver.mu_w(self.w, emp_p_in, two_players, alpha, no_one_hot=no_one_hot, fix_horizon = fix_horizon)
                policy = alpha * max_ent_policy + (1-alpha) * opponent_policy
            else:
                max_ent_policy,  opponent_policy = self.solver.two_players_soft_Q(alpha = alpha, beta = beta, beta_op = beta_op, n_episodes = 1000, lr = softQ_lr, reuseQ=False)
                policy = alpha * max_ent_policy + (1-alpha) * opponent_policy
                if fix_horizon:
                    mu_learner = self.solver.mu_policy_fixed_horizon(policy, stochastic=True, emp_p_in=emp_p_in, no_one_hot = no_one_hot)
                else:
                    mu_learner = self.solver.mu_policy(policy, stochastic=True, emp_p_in=emp_p_in, no_one_hot = no_one_hot)

            # Error
            err_t = np.linalg.norm(self.mu_teacher - mu_learner)


            err.append(err_t)
            rewards.append(np.copy(self.w))
            player_policies.append(max_ent_policy)
            adv_policies.append(opponent_policy)
            policies.append(policy)


            if verbose:
                print("Step", opt.step, ", error : ", err_t)

            if np.linalg.norm(grad) < tol:
                break

            if((opt.step+1) % 5):
                with open(base_dir + '/policy_' + str(lr) , "wb") as fp:   #Pickling
                    pickle.dump(policies, fp)
                with open(base_dir + '/player_' + str(lr) , "wb") as fp:   #Pickling
                    pickle.dump(player_policies, fp)
                with open(base_dir + '/adv_' + str(lr) , "wb") as fp:   #Pickling
                    pickle.dump(adv_policies, fp)
                with open(base_dir + '/reward_' + str(lr) , "wb") as fp:   #Pickling
                    pickle.dump(rewards, fp)
                with open(base_dir +'/err_' + str(lr) , "wb") as fp:   #Pickling
                    pickle.dump(err, fp)

        return policies, player_policies, adv_policies, rewards, err, self.solver.v


    def deep_max_ent(self, base_dir, optimizer, reward_nn, p_initial, beta, beta_op, alpha, softQ_lr, IRL_lr=0.1, n_steps = 150,reg_opp = False, no_one_hot = True, fix_horizon = False):
        err = []
        policies = []
        player_policies = []
        adv_policies = []

        i = 0
        tol = 1e-10
        width = int(np.sqrt(self.solver.env.n_states))
        features_dim = self.solver.env.features.shape[1]
        num_features = self.solver.env.features.shape[0]
        while i < n_steps:
            #adjust_learning_rate(IRL_lr, optimizer,i)
            features = self.solver.env.features
            reward = reward_nn(torch.from_numpy(features.reshape(1, features_dim , width, width)).float())

            # update reward(state, action) table
            R = self.solver.env.compute_reward_update(reward.detach().numpy().reshape(num_features))

            if not reg_opp:
                player, adv = self.solver.soft_2player_value_iteration(alpha = alpha)
                policy = alpha * player + (1-alpha) * adv
                if fix_horizon:
                    mu_learner = self.solver.mu_policy_fixed_horizon(policy, stochastic=True, emp_p_in=p_initial, no_one_hot=no_one_hot)

                else: 
                    mu_learner = self.solver.mu_policy(policy, stochastic=True, emp_p_in=p_initial, no_one_hot=no_one_hot)
            else:
                player,  adv = self.solver.two_players_soft_Q(alpha = alpha, beta = beta, beta_op = beta_op, n_episodes = 1000, lr = softQ_lr, reuseQ=False)
                policy = alpha * player + (1-alpha) * adv
                if fix_horizon:
                    mu_learner = self.solver.mu_policy_fixed_horizon(policy, stochastic=True, emp_p_in=p_initial, no_one_hot=no_one_hot)

                else: 
                    mu_learner = self.solver.mu_policy(policy, stochastic=True, emp_p_in=p_initial, no_one_hot=no_one_hot)

            gradient = -(self.mu_teacher - mu_learner)
            optimizer.zero_grad()
            reward.backward(torch.FloatTensor(gradient.reshape(1, 1, width, width)), retain_graph=True)
            optimizer.step()

            err_t = np.linalg.norm(self.mu_teacher - mu_learner)
            err.append(err_t)
            policies.append(policy)
            player_policies.append(player)
            adv_policies.append(adv)

            print("Step", i, ", error : ", err_t)

            if np.linalg.norm(gradient) < tol:
                break
            i += 1
            if((i+1) % 5):
                neural_net_state_reward = reward.detach().numpy().reshape(self.solver.env.n_states)
                with open(base_dir + '/policy_' + str(IRL_lr) , "wb") as fp:   #Pickling
                    pickle.dump(policies, fp)
                with open(base_dir + '/player_' + str(IRL_lr) , "wb") as fp:   #Pickling
                    pickle.dump(player_policies, fp)
                with open(base_dir + '/adv_' + str(IRL_lr) , "wb") as fp:   #Pickling
                    pickle.dump(adv_policies, fp)
                with open(base_dir + '/reward_' + str(IRL_lr) , "wb") as fp:   #Pickling
                    pickle.dump(neural_net_state_reward, fp)
                with open(base_dir +'/err_' + str(IRL_lr) , "wb") as fp:   #Pickling
                    pickle.dump(err, fp)

        return policies, player_policies, adv_policies, neural_net_state_reward, err, self.solver.v

    
