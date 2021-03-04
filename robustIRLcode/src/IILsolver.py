import numpy as np
from optimizers import GDOptimizer, AdamOptimizer
from RewardNet import RewardNet,adjust_learning_rate
from plot import plot_on_grid, plot_value_and_policy
from agent import Agent
from utils import randomize_optimal_policy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import pickle

from heapdict import heapdict
from i2l.discriminator_model import Discriminator, trajectoryDiscriminator
from i2l.wcritic_model import Wcritic, trajectoryWcritic
from i2l.policy_net import PPO

class Buffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.h_state = heapdict()
        self.h_state_action = heapdict()

    def add(self, mu_state, mu_state_action, score):

        mu_state = np.expand_dims(mu_state, axis = 1)
        mu_state = tuple(map(tuple, mu_state))
        mu_state_action = tuple(map(tuple, mu_state_action))
        print(len(self.h_state), "LEN h state")
        print(self.capacity, "CAPACITY")
        if (len(self.h_state) < self.capacity - 1):
            lowest_priority = -np.inf
        else:
            print("PEEKING")
            (_, lowest_priority) = self.h_state.peekitem()

        if score > lowest_priority:
            print("ADD", lowest_priority)
            self.h_state[mu_state] = score
            self.h_state_action[mu_state_action] = score
            if (len(self.h_state) == self.capacity):
                _ = self.h_state.popitem()
                _ = self.h_state_action.popitem()

    def get_average_mu(self):
        mu_tot = 0
        for mu in self.h_state.keys():
            mu_tot += np.array(mu)
        return np.squeeze(mu_tot, axis = 1) / len(self.h_state)

    def get_average_mu_action(self):
        mu_tot = 0
        for mu in self.h_state_action.keys():
            mu_tot += np.array(mu)
        return mu_tot / len(self.h_state)

    def get_mus(self):
        mus = []
        for mu in self.h_state.keys():
            mu = np.squeeze(np.array(mu), 1)
            mus.append(mu)
        return mus

    def get_mus_actions(self):
        mus = []
        for mu in self.h_state_action.keys():
            mu = np.array(mu)
            mus.append(mu)
        return mus

    def is_empty(self):

        return (len(self.h_state) == 0)

class trajectoryBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.h_state = heapdict()
        self.h_state_svf = heapdict()
        self.h_action = heapdict()
        self.h_ids = heapdict()

    def add(self, states, states_svf, actions, policy_ids, score):
        policy_ids = np.expand_dims(policy_ids, axis = 1)
        states = tuple(map(tuple, states))
        states_svf = tuple(map(tuple, states_svf))
        actions = tuple(map(tuple, actions))
        policy_ids = tuple(map(tuple, policy_ids))
        if (len(self.h_state) < self.capacity - 1):
            lowest_priority = -np.inf
        else:
            (_, lowest_priority) = self.h_state.peekitem()

        if score > lowest_priority:

            if not (actions in self.h_action.keys() or states in self.h_state.keys()):
                print("ADD", lowest_priority)
                self.h_state[states] = score
                self.h_state_svf[states_svf] = score
                self.h_action[actions] = score
                self.h_ids[policy_ids] = score
                if (len(self.h_state) == self.capacity):
                    _ = self.h_state.popitem()
                if (len(self.h_state_svf) == self.capacity):
                    _ = self.h_state_svf.popitem()
                if (len(self.h_action) == self.capacity):
                    _ = self.h_action.popitem()
                if (len(self.h_ids) == self.capacity):
                    _ = self.h_ids.popitem()

    def get_average_mu(self):
        mu_tot = 0
        for mu in self.h_state.keys():
            mu_tot += np.array(mu).mean(0)
        return mu_tot / len(self.h_state)

    def get_average_mu_svf(self):
        mu_tot = 0
        for mu in self.h_state_svf.keys():
            mu_tot += np.array(mu).mean(0)
        return mu_tot / len(self.h_state_svf)

    """def get_average_mu_action(self):
        mu_tot = 0
        for mu in self.h_action.keys():
            mu_tot += np.array(mu)
        return mu_tot / len(self.h_state)"""

    def get_states(self):
        state_list = []
        for state in self.h_state.keys():
            state_list.append(np.array(state))
        return state_list

    def get_ids(self):
        ids_list = []
        for ids in self.h_ids.keys():
            ids_list.append(np.array(ids))
        return ids_list

    def get_actions(self):
        action_list = []
        for action in self.h_action.keys():
            action = np.array(action)
            action_list.append(action)
        return action_list

    def is_empty(self):

        return (len(self.h_state) == 0)
"""
class NetworksManager:

    def __init__(self, demon_states_teacher, solver):
        obs_dim = solver.env.n_states
        acs_dim = solver.env.n_actions
        self.solver = solver
        self.discriminator = Discriminator(obs_dim, acs_dim, hidden_dim=64)
        # Wass-critic to discriminate b/w "state-only" data from the pq-buffer and the expert data
        self.wcritic = Wcritic(obs_dim, hidden_dim=64)

        self.buffer = Buffer(capacity = 5)
        # high level wrapper around a class that can manage multiple priority queues (if needed)
        self.expert_trajs = demon_states_teacher
        # (infinite) generator to loop over expert states
        self.optim_params = {   "airl_grad_steps" : 5,
                                "wcritic_grad_steps": 20,
                                "expert_batch_size" : 500}

    def update(self, iter_num, learner_demon_states, learner_demon_actions):
        if not self.super_pq.is_empty:
            # perform multiple updates of the wcritic classifier using pq-buffer and expert data
            wcritic_loss = self.wcritic.update(iter_num, self.expert_trajs, self.super_pq.random_select(),
                    batch_size=self.optim_params['expert_batch_size'], num_grad_steps=self.optim_params['wcritic_grad_steps'])
        else: wcritic_loss = 0.

        completed_trajs_scores = self.wcritic.assign_score(learner_demon_states)



        # Update the entries in the pq-buffers using the latest critic
        if iter_num and iter_num % self.super_pq.refresh_rate == 0:
            self.super_pq.update()

        # Update rewards with values from the discriminator
        self.discriminator.predict_batch_rewards(learner_demon_states, learner_demon_actions)

        # Perform multiple updates of the discriminator classifier using rollouts and pq-buffer
        discriminator_loss = self.discriminator.update(self.solver.policy, self.super_pq.random_select(),
                learner_demon_states, learner_demon_actions, num_grad_steps=self.optim_params['airl_grad_steps'])

        return [wcritic_loss, discriminator_loss]

"""


class IILsolver:
    def __init__(self, solver, mu_teacher, policy_in = None, w_in=None, buffer_capacity = 5):
        self.solver = solver
        self.mu_teacher = mu_teacher
        self.solver.policy = self.solver.env.random_policy() if policy_in is None else policy_in

        if w_in is None:
            w_in = -np.ones(self.solver.env.features_dim)
        self.w = w_in
        self.n_features = self.solver.env.features.shape[1]
        self.buffer = Buffer(buffer_capacity)
        self.discriminator = Discriminator(self.n_features, self.solver.env.n_actions, 20, self.solver.env) #TODO Check parameters
        self.wcritic = Wcritic( self.n_features,
                                20,
                                self.solver.env)
        self.PPO = PPO(self.solver.env)
        self.optim_params = {   "airl_grad_steps" : 5,
                                "wcritic_grad_steps": 20,
                                "expert_batch_size" : 500}


    def i2l(self,
            base_dir,
            tol=1e-10,
            verbose=True,
            lr=0.1,
            emp_p_in = None,
            n_steps = 200,
            no_one_hot = True):
        #Agent initialized with a random policy
        policies = []
        rewards = []
        err = []

        # update policy using value iteration
        #self.solver.value_iteration()
        # Apply arbitrary tie-breaking
        #self.solver.policy = randomize_optimal_policy(self.solver)
        #self.solver.policy = self.solver.soft_value_iteration()
        self.PPO.run()
        self.solver.policy, value_function = self.PPO.get_policy()
        old_policy = np.copy(self.solver.policy)
        for j in range(n_steps):

            rewards.append(np.copy(self.solver.env.r))
            policies.append(self.solver.policy)

            mu_learner = self.solver.mu_policy( self.solver.policy,
                                                stochastic = True, emp_p_in=emp_p_in,
                                                no_one_hot = no_one_hot)
            err_t = self.mu_teacher - mu_learner                                 
            err.append(np.linalg.norm(err_t))
            # Compute the state action occuoancy measure for the learner
            mu_learner_actions = mu_learner.reshape(-1,1).repeat(self.solver.env.n_actions, axis = 1) * self.solver.policy

            if not self.buffer.is_empty():
                # perform multiple updates of the wcritic classifier using pq-buffer and expert data
                wcritic_loss = self.wcritic.update( j,
                                                    self.mu_teacher,
                                                    self.buffer,
                                                    batch_size=self.optim_params['expert_batch_size'],
                                                    num_grad_steps=self.optim_params['wcritic_grad_steps'])
            else: wcritic_loss = 0.

            # Assign a score to the current learner state occupancy measure
            # mu_learner_score = self.wcritic.assign_score(mu_learner)
            mu_learner_score = self.wcritic._single_mu_score(mu_learner)
            # Add the mu to the buffer if the priority score is high enough
            self.buffer.add(mu_learner, mu_learner_actions, mu_learner_score)
            
            # Update the discriminator parameter
            discriminator_loss = self.discriminator.update(self.solver.policy, self.buffer,
                mu_learner_actions, num_grad_steps=self.optim_params['airl_grad_steps'])
            # Update the state action reward table r of shape (n_states, n_actions)
            self.solver.env.r = self.discriminator.predict_batch_rewards(mu_learner_actions)
            #print(self.solver.env.r, "REWARD")
            if j % 2 == 0:
                import matplotlib.pyplot as plt
                assert((self.solver.policy == old_policy).all())
                plot_value_and_policy(self.solver, self.solver.policy, "1", mode = "multiple", show = True)
                plot_on_grid(self.buffer.get_average_mu(), size = self.solver.env.size, title = "aa")
            # update policy using value iteration
            # self.solver.value_iteration()
            # Apply arbitrary tie-breaking
            #self.solver.policy = randomize_optimal_policy(self.solver)
            #self.solver.policy = self.solver.soft_value_iteration()
            self.PPO.run()
            self.solver.policy, value_function = self.PPO.get_policy()
            old_policy = np.copy(self.solver.policy)
            if j % 2:
                plot_on_grid(value_function, size = self.solver.env.size, title="critic value")
            
            if verbose:
                print(  "Step", j, ", error : ", np.linalg.norm(err_t), ", \
                        discriminator loss : ", discriminator_loss, ", \
                        wcritic loss : ", wcritic_loss)

            if np.linalg.norm(err_t) < tol:
                break

            if((j+1) % 5):
                with open(base_dir + '/policy_' + str(lr) , "wb") as fp:   #Pickling
                    pickle.dump(policies, fp)
                with open(base_dir + '/reward_' + str(lr) , "wb") as fp:   #Pickling
                    pickle.dump(rewards, fp)
                with open(base_dir +'/err_' + str(lr) , "wb") as fp:   #Pickling
                    pickle.dump(err, fp)

        # return policies, rewards, err, self.solver.v


class trajectoryIILsolver:
    #TODO: Try to take as input teacher trajectories
    def __init__(self, solver, teacher_states, policy_in = None, w_in=None, buffer_capacity =50, no_one_hot=True):
        self.solver = solver
        mu_teacher = []
        mu_svf_teacher = []
        for item in teacher_states: #for each trajectory
            for state in item:
                mu_teacher.append(self.solver.env.get_features(state = state))
                mu_svf_teacher.append(self.solver.env.get_svf_features(state = state))  
        goal_state = teacher_states[0][-1]
        self.mu_teacher = np.array(mu_teacher).mean(0)
        self.mu_svf_teacher = np.array(mu_svf_teacher).mean(0)
        self.solver.policy = self.solver.env.random_policy() if policy_in is None else policy_in

        if w_in is None:
            w_in = -np.ones(self.solver.env.features_dim)
        self.w = w_in
        self.n_features = self.solver.env.features.shape[1]
        print(self.n_features, "shape")
        self.buffer = trajectoryBuffer(buffer_capacity)
        self.discriminator = trajectoryDiscriminator(self.n_features, self.solver.env.n_actions, 20, self.solver.env) #TODO Check parameters
        self.wcritic = trajectoryWcritic( self.n_features,
                                20,
                                self.solver.env)
        print(solver.env.n_states, "n_states")
        self.PPO = PPO(self.solver.env, goal_state = goal_state)
        self.optim_params = {   "airl_grad_steps" : 5,
                                "wcritic_grad_steps": 20,
                                "expert_batch_size" : 500}


    def i2l(self,
            base_dir,
            tol=1e-10,
            verbose=True,
            lr=0.1,
            emp_p_in = None,
            n_steps = 200,
            no_one_hot = True):
        #Agent initialized with a random policy
        policies = []
        rewards = []
        err = []

        # update policy using value iteration
        #self.solver.value_iteration()
        # Apply arbitrary tie-breaking
        #self.solver.policy = randomize_optimal_policy(self.solver)
        #self.solver.policy = self.solver.soft_value_iteration()
        self.PPO.run(episodes = 1)
        self.solver.policy, value_function = self.PPO.get_policy()
        old_policy = np.copy(self.solver.policy)
        for j in range(n_steps):

            rewards.append(np.copy(self.solver.env.r))
            policies.append(self.solver.policy)

            mu_learner = self.solver.mu_policy( self.solver.policy,
                                                stochastic = True, emp_p_in=emp_p_in,
                                                no_one_hot = no_one_hot)
            mu_svf_learner = self.solver.mu_policy( self.solver.policy,
                                                    stochastic = True, emp_p_in=emp_p_in,
                                                    no_one_hot = False)
            err_t = self.mu_teacher - mu_learner                                 
            err.append(np.linalg.norm(err_t))

            # Compute the state action occuoancy measure for the learner
            mu_learner_actions = mu_svf_learner.reshape(-1,1).repeat(self.solver.env.n_actions, axis = 1) * self.solver.policy

            if not self.buffer.is_empty():
                # perform multiple updates of the wcritic classifier using pq-buffer and expert data
                wcritic_loss = self.wcritic.update( j,
                                                    self.mu_svf_teacher,
                                                    self.buffer,
                                                    batch_size=self.optim_params['expert_batch_size'],
                                                    num_grad_steps=self.optim_params['wcritic_grad_steps'])
            else: wcritic_loss = 0.

            # Assign a score to the current learner state occupancy measure
            # mu_learner_score = self.wcritic.assign_score(mu_learner)
            learner = Agent(self.solver.env, policy = self.solver.policy)

            state_trajs, actions, _, _, _ = learner.collect_trajectories(n_traj = 100, limit = 1000)
            state_ids = []
            is_to_delete = []
            for i, state_traj in enumerate(state_trajs):
                if len(state_traj) > 1:
                    state_ids.append([])
                    for state in state_traj:
                        #print(state, "STATE")
                        state_ids[-1].append(self.solver.env.state_to_index(state))
                else:
                    is_to_delete.append(i)
            #print(is_to_delete)
            for ii in reversed(is_to_delete):
                #state_trajs = list(state_trajs)
                actions = list(actions)
                #del state_trajs[ii]
                del actions[ii]
                #state_trajs = np.array(state_trajs)
                actions = np.array(actions)
            #print(len(state_ids[0]), "STATE_IDS")
            #print(len(actions[0]), "ACS")
            scores = self.wcritic.assign_score_trajs(state_ids)
            # Add the mu to the buffer if the priority score is high enough
            state_matrices = []
            state_svf_matrices = []
            action_matrices = []
            action_matrix = np.eye(self.solver.env.n_actions)
            policy_ids = []
            for i, trajs in enumerate(zip(state_ids, actions)):
                states = []
                states_svf = []
                acs = []
                if len(trajs[0]) > 1:
                    for state_id in trajs[0]:
                        states.append(self.solver.env.get_features(state_id = state_id))
                        states_svf.append(self.solver.env.get_svf_features(state_id = state_id))
                    for a in trajs[1]:
                        acs.append(action_matrix[a])
                    policy_ids.append([])
                    for k, state_id in enumerate(trajs[0][:-1]):
                        #for a in trajs[1]:
                        policy_ids[-1].append(state_id*self.solver.env.n_actions + trajs[1][k])    
                    states_inp = np.array(states)
                    states_svf_inp = np.array(states_svf)
                    acs_inp = np.array(acs)
                    state_matrices.append(states_inp)
                    state_svf_matrices.append(states_svf_inp)
                    action_matrices.append(acs_inp)
            for i, score in enumerate(scores):
                #print(state_matrices[i].shape, "STATE ADDING")
                #print(action_matrices[i].shape, "ACTIONS ADDING")
                self.buffer.add(state_matrices[i], state_svf_matrices[i], action_matrices[i], np.array(policy_ids[i]), score)
            
            # Update the discriminator parameter
            discriminator_loss = self.discriminator.update(self.solver.policy, self.buffer,
                state_matrices, action_matrices, policy_ids, num_grad_steps=self.optim_params['airl_grad_steps'])
            # Update the state action reward table r of shape (n_states, n_actions)
            self.solver.env.r = self.discriminator.predict_batch_rewards(mu_learner_actions)
            #print(self.solver.env.r, "REWARD")
            if j % 2 == 0:
                import matplotlib.pyplot as plt
                assert((self.solver.policy == old_policy).all())
                plot_value_and_policy(self.solver, self.solver.policy, "1", mode = "max_ent", show = True)
                #plot_on_grid(self.buffer.get_average_mu(), size = self.solver.env.size, title = "aa")
            # update policy using value iteration
            #self.solver.value_iteration()
            # Apply arbitrary tie-breaking
            #self.solver.policy = randomize_optimal_policy(self.solver)
            #self.solver.policy = self.solver.soft_value_iteration()
            self.PPO.run()
            self.solver.policy, value_function = self.PPO.get_policy()
            old_policy = np.copy(self.solver.policy)
            #print(self.solver.policy, "PPO policy")
            if j % 2:
                plot_on_grid(value_function, size = self.solver.env.size, title="critic value")
            if verbose:
                print(  "Step", j, ", error : ", np.linalg.norm(err_t), ", \
                        discriminator loss : ", discriminator_loss, ", \
                        wcritic loss : ", wcritic_loss)

            if np.linalg.norm(err_t) < tol:
                break

            if((j+1) % 5):
                with open(base_dir + '/policy_' + str(lr) , "wb") as fp:   #Pickling
                    pickle.dump(policies, fp)
                with open(base_dir + '/reward_' + str(lr) , "wb") as fp:   #Pickling
                    pickle.dump(rewards, fp)
                with open(base_dir +'/err_' + str(lr) , "wb") as fp:   #Pickling
                    pickle.dump(err, fp)