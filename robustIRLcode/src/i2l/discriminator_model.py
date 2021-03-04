import torch
import torch.nn as nn
import numpy as np
from utils import log_sum_exp

class Discriminator(nn.Module):
    def __init__(self, ob_dim, ac_dim, hidden_dim, env):
        super(Discriminator, self).__init__()

        self.ob_dim = ob_dim
        self.ac_dim = ac_dim
        input_dim = ob_dim + ac_dim

        self.env = env

        actv = nn.Tanh
        self.tower = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), actv(),
            #nn.Linear(hidden_dim, hidden_dim), actv(),
            nn.Linear(hidden_dim, 1, bias=False))

        # log(normalization-constant)
        self.logZ = nn.Parameter(torch.ones(1))
        
        self.action_one_hot = np.eye(ac_dim)
        input = []
        for n in range(env.n_states):
            for a in range(env.n_actions):
                # print(env.get_features(state_id = n).shape)
                # print(env.actions[a].shape)
                #input.append(np.concatenate((env.get_features(state_id = n), env.actions[a])))
                input.append(np.concatenate((env.get_features(state_id = n), self.action_one_hot[a])))
        inputs = np.array(input)
        self.inputs = torch.FloatTensor(inputs)
        #std = self.inputs.std(0)
        #print(std.shape)
        #print(self.inputs.shape)
        #self.inputs = (self.inputs - self.inputs.mean(0))/std
        #print(self.inputs.shape, "SHAPE")
        self.optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        # self.device = device
        # self.to(device)
        self.train()

    def forward(self):
        raise NotImplementedError()

    def update(self, policy, buffer, mu_learner_actions, num_grad_steps):
        self.train()
        loss_val = 0
        n = 0
        epsilon_policy = 1e-5
        mu_buffer_actions = torch.FloatTensor(buffer.get_average_mu_action()).view(self.env.n_states*self.env.n_actions, 1)
        mu_learner_actions = torch.FloatTensor(mu_learner_actions).view(self.env.n_states*self.env.n_actions, 1)
        self.policy_torch = torch.FloatTensor((policy + epsilon_policy)/(1 + epsilon_policy))
        self.log_probs = torch.log(self.policy_torch).view(self.env.n_states*self.env.n_actions, 1) ## self.q
        print(np.linalg.norm(mu_buffer_actions - mu_learner_actions), "Buffer-Learner Difference")
        for _ in range(num_grad_steps):

            """logp = self.tower(self.inputs)
            logq = self.log_probs + self.logZ.expand_as(self.log_probs)
            
            log_pq = torch.cat([logp, logq], dim=1)
            log_pq = log_sum_exp(log_pq, dim=1, keepdim=True)

            policy_loss = -(logq - log_pq) #.mean(0)
            pqb_loss = -(logp - log_pq) #.mean(0)

            pqb_out = torch.matmul(mu_buffer_actions.T, pqb_loss)
            learner_out = torch.matmul(mu_learner_actions.T, policy_loss)
            reward_bias = (-torch.cat([pqb_out, learner_out], dim=0)).clamp_(min=0).mean(0)
            loss = pqb_out + learner_out + 2*reward_bias"""
            """f = self.tower(self.inputs)
            exp_f = torch.exp(f)
            buffer_term = exp_f/ (exp_f + self.policy_torch.view(self.env.n_states*self.env.n_actions, 1))
            learner_term = 1 - buffer_term

            buffer_out = torch.matmul(mu_buffer_actions.T, torch.log(buffer_term))
            learner_out = torch.matmul(mu_learner_actions.T, torch.log(learner_term))

            loss = -(buffer_out + learner_out)"""
            f = self.tower(self.inputs)
            mixed_term = torch.log(torch.exp(f) + self.policy_torch.view(self.env.n_states*self.env.n_actions, 1))
            mu_sum = mu_learner_actions + mu_buffer_actions
            loss = - torch.matmul(mu_buffer_actions.T, f) - torch.matmul(mu_learner_actions.T, self.log_probs) + torch.matmul(mu_sum.T, mixed_term)
            """policy_logp = self.tower(self.inputs)
            pqb_logp = self.tower(self.inputs)

            policy_logq = self.log_probs + self.logZ.expand_as(self.log_probs)
            pqb_logq = self.log_probs + self.logZ.expand_as(self.log_probs)
            
            policy_log_pq = torch.cat([policy_logp, policy_logq], dim=1)
            policy_log_pq = log_sum_exp(policy_log_pq, dim=1, keepdim=True)

            pqb_log_pq = torch.cat([pqb_logp, pqb_logq], dim=1)
            pqb_log_pq = log_sum_exp(pqb_log_pq, dim=1, keepdim=True)
            
            policy_loss = -(policy_logq - policy_log_pq) #.mean(0)
            pqb_loss = -(pqb_logp - pqb_log_pq) #.mean(0)

            pqb_out = torch.matmul(mu_buffer_actions.T, pqb_loss)
            learner_out = torch.matmul(mu_learner_actions.T, policy_loss)
            reward_bias = (-torch.cat([pqb_out, learner_out], dim=0)).clamp_(min=0).mean(0)
            loss = pqb_out + learner_out + 2*reward_bias"""
            
            loss_val += loss.item()
            n += 1

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=10.)
            self.optimizer.step()
        
        return loss_val / n

    def predict_batch_rewards(self, mus):
        with torch.no_grad():
            self.eval()
            f = self.tower(self.inputs)
        f = f.detach().numpy().reshape(mus.shape)
        #log_probs = self.log_probs.detach().numpy().reshape(mus.shape)
        """exp_f = np.exp(f)
        probs = self.policy_torch.detach().numpy().reshape(mus.shape)
        D = exp_f/(exp_f + probs)
        rewards = np.log(D) - np.log(1 - D)
        return rewards #mus*rewards #- log_probs"""
        return f #- 0.001*log_probs

class trajectoryDiscriminator(nn.Module):
    def __init__(self, ob_dim, ac_dim, hidden_dim, env):
        super(trajectoryDiscriminator, self).__init__()

        self.ob_dim = ob_dim
        self.ac_dim = ac_dim
        input_dim = ob_dim + ac_dim

        self.env = env

        actv = nn.Tanh
        self.tower = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), actv(),
            #nn.Linear(hidden_dim, hidden_dim), actv(),
            nn.Linear(hidden_dim, 1, bias=False))

        # log(normalization-constant)
        self.logZ = nn.Parameter(torch.ones(1))
        self.action_one_hot = np.eye(ac_dim)
        input = []
        for n in range(env.n_states):
            for a in range(env.n_actions):
                # print(env.get_features(state_id = n).shape)
                # print(env.actions[a].shape)
                #input.append(np.concatenate((env.get_features(state_id = n), env.actions[a])))
                input.append(np.concatenate((env.get_features(state_id = n), self.action_one_hot[a])))
        inputs = np.array(input)
        self.inputs = torch.FloatTensor(inputs)
        #std = self.inputs.std(0)
        #print(std.shape)
        #print(self.inputs.shape)
        #self.inputs = (self.inputs - self.inputs.mean(0))/std
        #print(self.inputs.shape, "SHAPE")
        self.optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        # self.device = device
        # self.to(device)
        self.train()

    def forward(self):
        raise NotImplementedError()

    def update(self, policy, buffer, learner_states, learner_actions, learner_ids, num_grad_steps):
        self.train()
        loss_val = 0
        n = 0
        epsilon_policy = 1e-5
        #mu_learner_actions = torch.FloatTensor(mu_learner_actions).view(self.env.n_states*self.env.n_actions, 1)
        self.policy_torch = torch.FloatTensor((policy + epsilon_policy)/(1 + epsilon_policy))
        self.log_probs = torch.log(self.policy_torch).view(self.env.n_states*self.env.n_actions, 1) ## self.q
        self.state_buffer = buffer.get_states()
        self.action_buffer = buffer.get_actions()
        #print(np.linalg.norm(mu_buffer_actions - mu_learner_actions), "Buffer-Learner Difference")
        buffer_input = np.zeros((1, self.ob_dim + self.ac_dim))
        
        for item in zip(self.state_buffer, self.action_buffer):
            #print(item[0].shape, "STATE SHAPE")
            #print(item[1].shape, "ACTION SHAPE")
            partial_input = np.concatenate((item[0][:-1], item[1]), axis=1)
            buffer_input = np.vstack((buffer_input, partial_input))

        buffer_input = buffer_input[1:]
        buffer_input = torch.FloatTensor(buffer_input)
        learner_input = np.zeros((1, self.ob_dim + self.ac_dim))
        for item in zip(learner_states,learner_actions):
            partial_input = np.concatenate((item[0][:-1], item[1]), axis=1)
            learner_input = np.vstack((learner_input, partial_input))
            
        learner_input = learner_input[1:]
        learner_input = torch.FloatTensor(learner_input)
        buffer_policy_ids = buffer.get_ids()

        buffer_ids = []
        #print(len(buffer_policy_ids), "POLICY_IDS")
        #print(len(self.state_buffer))
        for trajs_id in buffer_policy_ids:
            for index in trajs_id:
                buffer_ids.append(index[0])
        #print(len(buffer_ids), "pre pytorch")
        buffer_log_probs = self.log_probs[buffer_ids]
        #print(len(buffer_log_probs), "post pytorch")
        l_ids = []
        for trajs_id in learner_ids:
            for index in trajs_id:
                l_ids.append(index)
        learner_log_probs = self.log_probs[l_ids]
        #print(buffer_log_probs.shape, "Buffer Policy")
        #print(buffer_input.shape, "Buffer input")
        for _ in range(num_grad_steps):
            
            buffer_logp = self.tower(buffer_input)
            learner_logp = self.tower(learner_input)

            learner_logq = learner_log_probs + self.logZ.expand_as(learner_log_probs)
            buffer_logq = buffer_log_probs + self.logZ.expand_as(buffer_log_probs)

            learner_log_pq = torch.cat([learner_logp, learner_logq], dim=1)
            learner_log_pq = log_sum_exp(learner_log_pq, dim=1, keepdim=True)

            buffer_log_pq = torch.cat([buffer_logp, buffer_logq], dim=1)
            buffer_log_pq = log_sum_exp(buffer_log_pq, dim=1, keepdim=True)

            learner_loss = -(learner_logq - learner_log_pq).mean(0)
            buffer_loss = -(buffer_logp - buffer_log_pq).mean(0)

            reward_bias = (-torch.cat([learner_logp, buffer_logp], dim=0)).clamp_(min=0).mean(0)
            loss = buffer_loss + learner_loss + 2*reward_bias
            
            
            loss_val += loss.item()
            n += 1

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=10.)
            self.optimizer.step()
        
        return loss_val / n

    def predict_batch_rewards(self, mus):
        with torch.no_grad():
            self.eval()
            f = self.tower(self.inputs)
        f = f.detach().numpy().reshape(mus.shape)
        #log_probs = self.log_probs.detach().numpy().reshape(mus.shape)
        """exp_f = np.exp(f)
        probs = self.policy_torch.detach().numpy().reshape(mus.shape)
        D = exp_f/(exp_f + probs)
        rewards = np.log(D) - np.log(1 - D)
        return rewards #mus*rewards #- log_probs"""
        return f