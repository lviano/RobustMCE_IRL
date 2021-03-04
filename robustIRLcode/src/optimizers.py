#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

class GDOptimizer:
    def __init__(self, lr, lr_order=1):
        self.type = "gd"

        self.lr = lr
        self.lr_order = lr_order

        self.step = 0
        self.eta = self.lr

    def update(self, grad):
        self.step += 1
        self.eta = self.lr*grad/pow(self.step, self.lr_order)
        return self.eta

class AdamOptimizer:
    def __init__(self, size, lr=0.5, b1=.9, b2=.99, epsilon=10e-8):
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

