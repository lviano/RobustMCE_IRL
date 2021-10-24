import pickle
import sys
import argparse
import copy
import ast
import os
import gym
import gym_simple
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from itertools import count
from utils import *
from utils import plot
from gail.value_iteration_gym import PolicyNet, value_iteration

def evaluate_loop(policy_net, running_state, expert_flag, env, is_disc_action, args):

    num_steps = 0
    episodes_reward = []
    dtype = torch.float64
    for i_episode in count():

        state = env.reset()
        state_expert = copy.deepcopy(state)
        """if args.env_name == "gridworld-v0":
            s_index_expert = env.state_to_index(state_expert)"""
        state = running_state(state)
        reward_episode = 0

        for t in range(10000):
            state_var = tensor(state).unsqueeze(0).to(dtype)
            # choose mean action
            if expert_flag and args.env_name == "gridworld-v0":
                """if args.env_name == "gridworld-v0":
                    action = policy_net.select_action(s_index_expert)
                else:"""
                action = policy_net.select_action(state_expert)
            else:
                if not is_disc_action:
                    action = policy_net(state_var)[0][0].detach().numpy()
                else:
                    action = policy_net.select_action(state_var)[0].numpy()
            # action = policy_net.select_action(state_var)[0].cpu().numpy()
            action = int(action) if is_disc_action else action.astype(np.float64)
            #print(action, "action")
            next_state, reward, done, _ = env.step(action.clip(-1.0, 1.0))
            state_expert = copy.deepcopy(next_state)
            """if args.env_name == "gridworld-v0":
                s_index_expert = env.state_to_index(state_expert)"""
            next_state = running_state(next_state)
            reward_episode += reward
            num_steps += 1


            if args.render:
                env.render()
            if done or num_steps >= args.max_steps:

                break

            state = next_state
        if done:
            episodes_reward.append(reward_episode)
        print('Episode {}\t reward: {:.2f}'.format(i_episode, reward_episode))

        if num_steps >= args.max_steps:
            break

    return np.mean(episodes_reward)
