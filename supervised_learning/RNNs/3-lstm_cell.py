#!/usr/bin/env python3
""" LSTM Cell implementation in Python """

import numpy as np


class LSTMCell:
    """Represents an LSTM unit."""

    def __init__(self, i, h, o):
        """
        Constructor for the LSTMCell class.
        """
        self.Wf = np.random.randn(i + h, h)
        self.Wu = np.random.randn(i + h, h)
        self.Wc = np.random.randn(i + h, h)
        self.Wo = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)

        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def sigmoid(self, z):
        """
        Sigmoid activation function.
        """
        return 1 / (1 + np.exp(-z))

    def softmax(self, z):
        """
        Softmax activation function.
        """
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, h_prev, c_prev, x_t):
        """
        Forward propagation for one time step.
        """
        m, i = x_t.shape
        h = h_prev.shape[1]

        # Concatenate x_t and h_prev to form input for gates
        concat_x_h = np.concatenate((h_prev, x_t), axis=1)

        # Forget gate
        ft = self.sigmoid(np.dot(concat_x_h, self.Wf) + self.bf)

        # Update gate
        ut = self.sigmoid(np.dot(concat_x_h, self.Wu) + self.bu)

        # Intermediate cell state
        c_hat = np.tanh(np.dot(concat_x_h, self.Wc) + self.bc)

        # Next cell state
        c_next = ft * c_prev + ut * c_hat

        # Output gate
        ot = self.sigmoid(np.dot(concat_x_h, self.Wo) + self.bo)

        # Next hidden state
        h_next = ot * np.tanh(c_next)

        # Output
        y = self.softmax(np.dot(h_next, self.Wy) + self.by)

        return h_next, c_next, y
