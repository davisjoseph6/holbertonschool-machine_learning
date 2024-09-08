#!/usr/bin/env python3
""" Bidirectional RNN Cell """

import numpy as np


class BidirectionalCell:
    """Represents a bidirectional cell of an RNN."""

    def __init__(self, i, h, o):
        """
        Class constructor that initializes the weights and biases.
        """
        self.Whf = np.random.randn(i + h, h)
        self.Whb = np.random.randn(i + h, h)
        self.Wy = np.random.randn(2 * h, o)

        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Calculates the hidden state in the forward direction for one time step.
        """
        # Concatenate the previous hidden state and input data
        concatenated = np.concatenate((h_prev, x_t), axis=1)

        # Compute the next hidden state using the forward weights and biases
        h_next = np.tanh(np.dot(concatenated, self.Whf) + self.bhf)

        return h_next

    def backward(self, h_next, x_t):
        """
        Calculates the hidden state in the backward direction for
        one time step.
        """
        concatenated = np.concatenate((h_next, x_t), axis=1)
        h_prev = np.tanh(np.dot(concatenated, self.Whb) + self.bhb)
        return h_prev

    def output(self, H):
        """
        Calculates all outputs for the RNN.
        """
        t, m, _ = H.shape
        Y = np.dot(H, self.Wy) + self.by
        Y = np.exp(Y) / np.sum(np.exp(Y), axis=2, keepdims=True)
        return Y
