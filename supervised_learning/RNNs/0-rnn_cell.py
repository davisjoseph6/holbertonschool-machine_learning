#!/usr/bin/env python3

import numpy as np


class RNNCell:
    """
    Represents a cell of a simple Recurrent Neural Network (RNN).
    """

    def __init__(self, i, h, o):
        """
        Initialize the RNNCell with given dimensions for input, hidden state,
        and output.
        """
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Performs forward propagation for one time step.
        """
        concatenated = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.dot(concatenated, self.Wh) + self.bh)
        y = self.softmax(np.dot(h_next, self.Wy) + self.by)
        return h_next, y

    @staticmethod
    def softmax(x):
        """
        Applies the softmax activation function.
        """
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / exp_x.sum(axis=1, keepdims=True)
