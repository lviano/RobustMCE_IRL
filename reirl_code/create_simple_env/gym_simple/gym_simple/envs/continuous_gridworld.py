import math
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
import copy
import random
from scipy import sparse

class TabularEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }


    def __init__(self, prop):

        self.viewer = None

        self.reward_range = (-100, 0)
        # although there are 2 terminal squares in the grid
        # they are considered as 1 state
        # therefore observation is between 0 and 14
        self.action_space = spaces.Box(low=-0.5,
                                       high=0.5,
                                       shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)



        # Discount factor
        self.gamma = 0.99

        # Stochastic transitions
        self.prop_random_actions = prop

class ContinuousGridWorld(TabularEnv):
    def __init__(self, env_type=0, prop=0):
        # Characteristics of the gridworld

        TabularEnv.__init__(self, prop)
        self.env_type = env_type
        self.state=None
        self.prop=prop
        self.env_type=env_type
        self.seed()
        self.reset()
        self.steps_from_last_reset=0
        if env_type == 0:
            self.terminal_area = np.array([[-1.0, -0.95], [0.95, 1.0]])
        else:
            self.terminal_area = np.array([[0.95, 1.0],[-1.0, -0.95]])

    def step(self, a):
        #print(a, "action")
        a[0] = np.max([np.min([0.5, a[0]]), -0.5])
        a[1] = np.max([np.min([0.5, a[1]]), -0.5])
        self.state += (1-self.prop)*a/10. + self.prop*np.random.uniform(-0.1, 0.1, size=2)
        self.state[0] = np.max([np.min([1,self.state[0]]),-1])
        self.state[1] = np.max([np.min([1, self.state[1]]), -1])
        if (self.terminal_area[0,0] <= self.state[0]  <= self.terminal_area[0,1] and
            self.terminal_area[1,0] <= self.state[1]  <= self.terminal_area[1,1]):
            self.done = True
        reward = self.compute_reward()
        self.steps_from_last_reset += 1
        if self.steps_from_last_reset == 5000:
            self.done = True
            return np.array([self.state[0],
                             self.state[1],
                             # self.state[0]*self.state[1],
                             self.state[0] ** 2,
                             self.state[1] ** 2,
                             # self.state[0] ** 3,
                             # self.state[1] ** 3,
                             (1 / (self.state[0] ** 2
                                            + self.state[1] ** 2
                                            + 1e-8)) ** 2,
                             0.0]
                            ), \
                   reward, \
                   self.done, \
                   None
        return np.array([self.state[0],
                        self.state[1],
                        #self.state[0]*self.state[1],
                        self.state[0]**2,
                        self.state[1]**2,
                        #self.state[0] ** 3,
                        #self.state[1] ** 3,
                        (1 / (self.state[0] ** 2
                               + self.state[1] ** 2
                               + 1e-8)) ** 2,
                        10.*float(self.done)]
                        ), \
               reward, \
               self.done, \
               None

    def reset(self, starting_index = None):
        if self.env_type == 0:
            self.state = np.random.uniform(-1, 1, size=2)
        else:
            self.state = np.array([-1.0, 1.0])
        self.steps_from_last_reset = 0
        self.done = False
        return np.array([self.state[0],
                        self.state[1],
                        #self.state[0]*self.state[1],
                        self.state[0]**2,
                        self.state[1]**2,
                        #self.state[0] ** 3,
                        #self.state[1] ** 3,
                        (1 / (self.state[0] ** 2
                               + self.state[1] ** 2
                               + 1e-8)) ** 2,
                        10.*float(self.done)]
                        )
    def compute_reward(self):
        if self.env_type==0:
            if (self.terminal_area[0,0] <= self.state[0]  <= self.terminal_area[0,1] and
            self.terminal_area[1,0] <= self.state[1]  <= self.terminal_area[1,1]):
                reward = -(self.state[0] ** 2 + self.state[1] ** 2) + 3 * \
                     self.state[0] - 5 + 2000
            else:
                reward = -(self.state[0] ** 2 + self.state[1] ** 2) + 3 * \
                     self.state[0] - 5
        elif self.env_type==1:
            reward = -(self.state[0] -1)**2 - (self.state[1] + 1)**2 - (1/(self.state[0]**2
                                                                 + self.state[1]**2
                                                                 + 1e-8))**2
            if (self.terminal_area[0, 0] <= self.state[0] <= self.terminal_area[
                0, 1] and
                    self.terminal_area[1, 0] <= self.state[1] <=
                    self.terminal_area[1, 1]):
                reward += 10
        return reward

class GaussianGridWorld(TabularEnv):
    def __init__(self, env_type=0, prop=0):
        # Characteristics of the gridworld

        TabularEnv.__init__(self, prop)
        self.env_type = env_type
        self.state=None
        self.prop=prop
        self.env_type=env_type
        self.seed()
        self.reset()
        self.steps_from_last_reset=0
        if env_type == 0:
            self.terminal_area = np.array([[-1.0, -0.95], [0.95, 1.0]])
        else:
            self.terminal_area = np.array([[0.95, 1.0],[-1.0, -0.95]])

    def step(self, a):
        #print(a, "action")
        a[0] = np.max([np.min([0.5, a[0]]), -0.5])
        a[1] = np.max([np.min([0.5, a[1]]), -0.5])
        #self.state += (1 - self.prop)*a/10. + self.prop*np.random.uniform(-0.1, 0.1, size=2)
        prob_adversarial_action = np.random.binomial(n=1, p=self.prop)
        if prob_adversarial_action:
            if self.env_type == 1:
                self.state += np.random.uniform(-0.1,0.1, size=2)
            elif self.env_type == 2:
                self.state -= 0.1*self.state/np.linalg.norm(self.state)
        else:
            self.state += a/10.
        #self.state = self.state + a/10. + self.prop*np.random.uniform(-0.1, 0.1, size=2)
        self.state[0] = np.max([np.min([1,self.state[0]]),-1])
        self.state[1] = np.max([np.min([1, self.state[1]]), -1])
        if (self.terminal_area[0,0] <= self.state[0]  <= self.terminal_area[0,1] and
            self.terminal_area[1,0] <= self.state[1]  <= self.terminal_area[1,1]):
            self.done = True
        reward = self.compute_reward()
        self.steps_from_last_reset += 1
        if self.steps_from_last_reset == 5000:
            self.done = True
            return np.array([self.state[0],
                             self.state[1],
                             # self.state[0]*self.state[1],
                             self.state[0] ** 2,
                             self.state[1] ** 2,
                             # self.state[0] ** 3,
                             # self.state[1] ** 3,
                             8*np.exp(-8*self.state[0]**2-8*self.state[1]**2),
                             0.0]
                            ), \
                   reward, \
                   self.done, \
                   None
        return np.array([self.state[0],
                        self.state[1],
                        #self.state[0]*self.state[1],
                        self.state[0]**2,
                        self.state[1]**2,
                        #self.state[0] ** 3,
                        #self.state[1] ** 3,
                        8*np.exp(-8*self.state[0]**2-8*self.state[1]**2),
                        10.*float(self.done)]
                        ), \
               reward, \
               self.done, \
               None

    def reset(self, starting_index = None):
        if self.env_type == 0:
            self.state = np.random.uniform(-1, 1, size=2)
        else:
            self.state = np.array([-1.0, 1.0])
        self.steps_from_last_reset = 0
        self.done = False
        return np.array([self.state[0],
                        self.state[1],
                        #self.state[0]*self.state[1],
                        self.state[0]**2,
                        self.state[1]**2,
                        #self.state[0] ** 3,
                        #self.state[1] ** 3,
                        8*np.exp(-8*self.state[0]**2-8*self.state[1]**2),
                        10.*float(self.done)]
                        )
    def compute_reward(self):
        if self.env_type==0:
            if (self.terminal_area[0,0] <= self.state[0]  <= self.terminal_area[0,1] and
            self.terminal_area[1,0] <= self.state[1]  <= self.terminal_area[1,1]):
                reward = -(self.state[0] ** 2 + self.state[1] ** 2) + 3 * \
                     self.state[0] - 5 + 2000
            else:
                reward = -(self.state[0] ** 2 + self.state[1] ** 2) + 3 * \
                     self.state[0] - 5
        elif self.env_type==1 or self.env_type==2:
            reward = -(self.state[0] -1)**2 - (self.state[1] + 1)**2 - 80*np.exp(-8*self.state[0]**2-8*self.state[1]**2)
            if (self.terminal_area[0, 0] <= self.state[0] <= self.terminal_area[
                0, 1] and
                    self.terminal_area[1, 0] <= self.state[1] <=
                    self.terminal_area[1, 1]):
                reward += 10
        return reward
