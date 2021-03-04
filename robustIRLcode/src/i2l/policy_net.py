import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import tensorboard

from MDPsolver import *
from plot import *

# helper function to convert numpy arrays to tensors
def t(x): return torch.from_numpy(x).float()

def mish(input):
    return input * torch.tanh(F.softplus(input))

class Mish(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, input): return mish(input)

class Actor(nn.Module):
    def __init__(self, state_dim, n_actions, activation=nn.Tanh):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            activation(),
            nn.Linear(64, 32),
            activation(),
            nn.Linear(32, n_actions),
            nn.Softmax()
        )
    
    def forward(self, X):
        return self.model(X)

# Critic module
class Critic(nn.Module):
    def __init__(self, state_dim, activation=nn.Tanh):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            activation(),
            nn.Linear(64, 32),
            activation(),
            nn.Linear(32, 1)
        )
    
    def forward(self, X):
        return self.model(X)

def clip_grad_norm_(module, max_grad_norm):
    nn.utils.clip_grad_norm_([p for g in module.param_groups for p in g["params"]], max_grad_norm)

def policy_loss(old_log_prob, log_prob, advantage, eps):
    ratio = (log_prob - old_log_prob).exp()
    clipped = torch.clamp(ratio, 1-eps, 1+eps)*advantage
    
    m = torch.min(ratio*advantage, clipped)
    return -m

class PPO:
    def __init__(self, environment, goal_state = [0, 0]):

        self.environment = environment
        self.state_dim = self.environment.n_states
        self.n_actions = self.environment.n_actions
        self.actor = Actor(self.state_dim, self.n_actions, activation=Mish)
        self.critic = Critic(self.state_dim, activation=Mish)
        self.adam_actor = torch.optim.Adam(self.actor.parameters(), lr= 3e-4)
        self.adam_critic = torch.optim.Adam(self.critic.parameters(), lr=1e-3)
        self.goal_state = goal_state
        torch.manual_seed(1)

    def run(self, episodes = 60):

        episode_rewards = []
        gamma = 0.98
        eps = 0.2 #0.4
        w = tensorboard.SummaryWriter()
        s = 0
        max_grad_norm = 0.5

        for i in range(episodes):
            prev_prob_act = None
            done = False
            total_reward = 0
            #state = env.reset()
            
            state_id = self.environment.get_random_initial_state()
            state = self.environment.get_svf_features(state=state_id)
            time = 0
            while (not done and time < 1000):
                time += 1
                s += 1
                probs = self.actor(t(state))
                dist = torch.distributions.Categorical(probs=probs)
                action = dist.sample()
                prob_act = dist.log_prob(action)
                #print(action.detach().numpy(), "ACTION")
                #print(state_id, "STATE")
                #next_state, reward, done, info = env.step(action.detach().data.numpy())
                next_state_id = self.environment.take_action(state_id, int(action.detach().numpy()))
                next_state = self.environment.get_svf_features(state=next_state_id)
                reward = self.environment.get_rewards(state=state_id)
                done = (state_id == self.goal_state).all()
                if done:
                    print(done, "DONE")
                #print(type(float(reward)), "REWARD")
                #print(torch.FloatTensor(np.array(reward)) )
                #print(critic(t(state)), "CRITIC STATE")
                #print(critic(t(next_state)), "CRITIC NEXT STATE")
                #print((1-done)*gamma*critic(t(next_state)) - critic(t(state)))
                advantage = float(reward) + (1-done)*gamma*self.critic(t(next_state)) - self.critic(t(state))
                #print(advantage, "ADVANTAGE")
                w.add_scalar("loss/advantage", advantage, global_step=s)
                w.add_scalar("actions/action_0_prob", dist.probs[0], global_step=s)
                w.add_scalar("actions/action_1_prob", dist.probs[1], global_step=s)
                w.add_scalar("actions/action_2_prob", dist.probs[2], global_step=s)
                w.add_scalar("actions/action_3_prob", dist.probs[3], global_step=s)
                total_reward += reward
                state = next_state
                state_id = next_state_id
                
                if prev_prob_act:
                    actor_loss = policy_loss(prev_prob_act.detach(), prob_act, advantage.detach(), eps)
                    w.add_scalar("loss/actor_loss", actor_loss, global_step=s)
                    self.adam_actor.zero_grad()
                    actor_loss.backward()
                    # clip_grad_norm_(adam_actor, max_grad_norm)
                    w.add_histogram("gradients/actor",
                                    torch.cat([p.grad.view(-1) for p in self.actor.parameters()]), global_step=s)
                    self.adam_actor.step()

                    critic_loss = advantage.pow(2).mean()
                    w.add_scalar("loss/critic_loss", critic_loss, global_step=s)
                    self.adam_critic.zero_grad()
                    critic_loss.backward()
                    # clip_grad_norm_(adam_critic, max_grad_norm)
                    w.add_histogram("gradients/critic",
                                    torch.cat([p.data.view(-1) for p in self.critic.parameters()]), global_step=s)
                    self.adam_critic.step()
                
                prev_prob_act = prob_act
            
            w.add_scalar("reward/episode_reward", total_reward, global_step=i)
            episode_rewards.append(total_reward)
            ### Plot policies after each episode
            #solver = MDPsolver(self.environment)
            #policy, _ = self.get_policy()
            #plot_value_and_policy(solver, policy, "1", mode = "max_ent", show = True)
            ###
    def get_policy(self):
        action_probs_list = []
        value_list = []
        with torch.no_grad():
            for s in range(self.state_dim):
                input = self.environment.get_svf_features(state_id=s)
                action_probs = self.actor(t(input)).detach().numpy()
                action_probs_list.append(action_probs)
                value_list.append(self.critic(t(input)).detach().numpy())
        return np.array(action_probs_list), np.array(value_list)

