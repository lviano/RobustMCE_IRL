import numpy as np
import torch
import copy

class Weights():
    def __init__(self, state_dim):
        self.weights = np.zeros(
            state_dim)  # -np.ones((state_dim))/state_dim #np.ones((1,state_dim))/state_dim

    def write(self, w):
        self.weights = copy.deepcopy(w)

    def read(self):
        return copy.deepcopy(self.weights)

class AdamOptimizer:
    def __init__(self, size, lr=0.5, b1=.9, b2=.99, epsilon=10e-6):
        self.type = "adam"

        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.epsilon = epsilon

        self.step = 0

        self.m = np.zeros(size)
        self.v = np.zeros(size)

    def update(self, grad):
        self.step += 1

        self.m = self.b1 * self.m + (1 - self.b1) * grad
        self.v = self.b2 * self.v + (1 - self.b2) * grad ** 2
        m_hat = self.m / (1 - self.b1 ** self.step)
        v_hat = self.v / (1 - self.b2 ** self.step)

        return (self.lr * m_hat) / (np.sqrt(v_hat) + self.epsilon)

