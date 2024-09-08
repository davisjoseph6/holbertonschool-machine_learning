#!/usr/bin/env python3
"""
This module contains the bi_rnn function for performing forward propagation
on a Bidirectional RNN.
"""

import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """
    Performs forward propagation for a bidirectional RNN.
    """
    t, m, i = X.shape
    h = h_0.shape[1]

    # Initialize arrays to store hidden states
    H_forward = np.zeros((t, m, h))
    H_backward = np.zeros((t, m, h))

    # Forward pass
    h_f = h_0
    for step in range(t):
        h_f = bi_cell.forward(h_f, X[step])
        H_forward[step] = h_f

    # Backward pass
    h_b = h_t
    for step in reversed(range(t)):
        h_b = bi_cell.backward(h_b, X[step])
        H_backward[step] = h_b

    # Concatenate forward and backward hidden states
    H = np.concatenate((H_forward, H_backward), axis=2)

    # Compute the output Y
    Y = bi_cell.output(H)

    return H, Y
