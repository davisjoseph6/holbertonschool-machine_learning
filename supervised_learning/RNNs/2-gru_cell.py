#!/usr/bin/env python3
"""
This module implements the GRUCell class, which represents a single unit of a
Gated Recurrent Unit (GRU) network.
"""

import numpy as np


class GRUCell:
    """
    Represents a single unit of a Gated Recurrent Unit (GRU) network.
    """

    def __init__(self, i, h, o):
        """
        Initializes the GRUCell with given dimensions for input, hidden state,
        and output.
        """
        self.Wz = np.random.randn(i + h, h)
        self.Wr = np.random.randn(i + h, h)
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Performs forward propagation for one time step.
        """
        concatenated = np.concatenate((h_prev, x_t), axis=1)

        # Update gate
        z_t = self.sigmoid(np.dot(concatenated, self.Wz) + self.bz)

        # Reset gate
        r_t = self.sigmoid(np.dot(concatenated, self.Wr) + self.br)

        # Intermediate hidden state
        concatenated_reset = np.concatenate((r_t * h_prev, x_t), axis=1)
        h_tilde = np.tanh(np.dot(concatenated_reset, self.Wh) + self.bh)

        # Next hidden state
        h_next = (1 - z_t) * h_prev + z_t * h_tilde

        # Output using softmax activation
        y = self.softmax(np.dot(h_next, self.Wy) + self.by)

        return h_next, y

    @staticmethod
    def sigmoid(x):
        """
        Applies the sigmoid activation function.
        """
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def softmax(x):
        """
        Applies the softmax activation function.
        """
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / exp_x.sum(axis=1, keepdims=True)
