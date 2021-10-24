import gym
import gym_simple
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import plot

grid_type = 1
dim = 5
env = gym.make("gridworld-v0", prop = 0.0, env_type = grid_type)
plot.plot_reward(env.state_r, dim, title="", tdw=False, show=True)




