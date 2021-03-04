import torch
import torch.nn as nn
import numpy as np

class Wcritic(nn.Module):
    def __init__(self, n_features, hidden_dim, env):
        super(Wcritic, self).__init__()

        actv = nn.Tanh
        self.tower = nn.Sequential(
            nn.Linear(n_features, hidden_dim), actv(),
            #nn.Linear(hidden_dim, hidden_dim), actv(),
            #nn.Linear(hidden_dim, hidden_dim), actv(),
            nn.Linear(hidden_dim, 1))

        self.warmup = 5
        self.clip = 0.05
        self.optimizer = torch.optim.RMSprop(self.parameters(), lr=5e-5)

        states = []
        for n in range(env.n_states):
            states.append(env.get_features(state_id = n))
        states = np.array(states)
        self.states_inp = torch.FloatTensor(states)
        self.env = env
        self.train()

    def forward(self):
        raise NotImplementedError()

    def update(self, n_iter, mu_teacher, buffer, batch_size, num_grad_steps):
        """
        Perform multiple updates of the wasserstein classifier using pq-buffer and expert data
        """
        self.train()

        # Compute the output of the network for each individual state

        if n_iter <= self.warmup:
            num_grad_steps *= (self.warmup + 1 - n_iter)

        loss_val = 0
        n = 0
        print(self.env.n_states, "n_states")
        mu_buffer = torch.FloatTensor(buffer.get_average_mu()).view(1,self.env.n_states)
        mu_teacher = torch.FloatTensor(mu_teacher).view(1,self.env.n_states)
        print(np.linalg.norm(mu_buffer - mu_teacher), "Difference Critic")
        for _ in range(num_grad_steps):

            out = self.tower(self.states_inp)

            pqb_out = torch.matmul(mu_buffer, out)
            teacher_out = torch.matmul(mu_teacher, out)
            reward_bias = - torch.clamp(pqb_out, max=0) - torch.clamp(teacher_out, max=0)
            loss = pqb_out - teacher_out + 2*reward_bias

            loss_val += loss.item()
            n += 1

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=10.)
            self.optimizer.step()

            # weight clamping to enforce the Lipchitz constraint
            for p in self.parameters():
                p.data.clamp_(-self.clip, self.clip)

        return loss_val / n

    def assign_score(self, mus):
        """
        Assign scores to occupancy measures
        """
        mus_scores = []
        with torch.no_grad():
            self.eval()
            out = self.tower(self.states_inp).detach().numpy()

        for mu in mus:
            mus_scores.append(mu.dot(out))

        return mus_scores #List of one dimensional arrays

    def _single_mu_score(self, mu):
        with torch.no_grad():
            self.eval()
            out = self.tower(self.states_inp).detach().numpy()
            return mu.dot(out)
    
    def assign_score_trajs(self, state_trajs): #state_trajs = #num_trajs x num_states
        scores = []
        for i, traj in enumerate(state_trajs[:-1]):

            states = []
            for state_id in traj:
                states.append(self.env.get_features(state_id = state_id))
            states_inp = np.array(states)
            print(states_inp.shape, "STATE_SHAPE")
            states_inp = torch.FloatTensor(states_inp)
            #print(self.tower(states_inp).mean(0).detach().numpy())
            with torch.no_grad():
                self.eval()
                scores.append(- self.tower(states_inp).mean(0).detach().numpy())
        return scores


class trajectoryWcritic(nn.Module):
    def __init__(self, n_features, hidden_dim, env):
        super(trajectoryWcritic, self).__init__()
        self.n_features = n_features
        actv = nn.Tanh
        self.tower = nn.Sequential(
            nn.Linear(n_features, hidden_dim), actv(),
            #nn.Linear(hidden_dim, hidden_dim), actv(),
            #nn.Linear(hidden_dim, hidden_dim), actv(),
            nn.Linear(hidden_dim, 1))

        self.warmup = 5
        self.clip = 0.05
        self.optimizer = torch.optim.RMSprop(self.parameters(), lr=5e-5)

        states = []
        for n in range(env.n_states):
            states.append(env.get_features(state_id = n))
        states = np.array(states)
        self.states_inp = torch.FloatTensor(states)
        self.env = env
        self.train()

    def forward(self):
        raise NotImplementedError()

    def update(self, n_iter, mu_teacher, buffer, batch_size, num_grad_steps):
        """
        Perform multiple updates of the wasserstein classifier using pq-buffer and expert data
        """
        self.train()

        # Compute the output of the network for each individual state

        if n_iter <= self.warmup:
            num_grad_steps *= (self.warmup + 1 - n_iter)

        loss_val = 0
        n = 0

        mu_buffer = torch.FloatTensor(buffer.get_average_mu_svf()).view(1, self.env.n_states)
        mu_teacher = torch.FloatTensor(mu_teacher).view(1, self.env.n_states)
        print(np.linalg.norm(mu_buffer - mu_teacher), "Difference Critic")
        for _ in range(num_grad_steps):

            out = self.tower(self.states_inp)

            pqb_out = torch.matmul(mu_buffer, out)
            teacher_out = torch.matmul(mu_teacher, out)
            reward_bias = - torch.clamp(pqb_out, max=0) - torch.clamp(teacher_out, max=0)
            loss = pqb_out - teacher_out + 2*reward_bias

            loss_val += loss.item()
            n += 1

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=10.)
            self.optimizer.step()

            # weight clamping to enforce the Lipchitz constraint
            for p in self.parameters():
                p.data.clamp_(-self.clip, self.clip)

        return loss_val / n

    def assign_score(self, mus):
        """
        Assign scores to occupancy measures
        """
        mus_scores = []
        with torch.no_grad():
            self.eval()
            out = self.tower(self.states_inp).detach().numpy()

        for mu in mus:
            mus_scores.append(mu.dot(out))

        return mus_scores #List of one dimensional arrays

    def _single_mu_score(self, mu):
        with torch.no_grad():
            self.eval()
            out = self.tower(self.states_inp).detach().numpy()
            return mu.dot(out)
    
    def assign_score_trajs(self, state_trajs): #state_trajs = #num_trajs x num_states
        scores = []
        for i, traj in enumerate(state_trajs[:-1]):

            states = []
            for state_id in traj:
                states.append(self.env.get_features(state_id = state_id))
            states_inp = np.array(states)
            print(states_inp.shape, "STATE_SHAPE")
            states_inp = torch.FloatTensor(states_inp)
            #print(self.tower(states_inp).mean(0).detach().numpy())
            with torch.no_grad():
                self.eval()
                scores.append(self.tower(states_inp).sum(0).detach().numpy())
        return scores