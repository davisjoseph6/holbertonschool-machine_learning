#!/usr/bin/env python3
"""
This module implements the function rnn that performs forward propagation
for a simple Recurrent Neural network (RNN) using an instance of the RNNCell
class.
"""

import numpy as np

def rnn(rnn_cell, X, h_0):
    """
    Performs forward propagation for a simple RNN.
    """
    t, m, i = X.shape
    _, h = h_0.shape

    # Initialize the array to store hidden states
    H = np.zeros((t + 1, m, h))
    H[0] = h_0

    # Initialize a list to store the outputs
    outputs = []

    # Perform forward propagation through each time step
    for step in range(t):
        h_prev = H[step]
        x_t = X[step]
        h_next, y = rnn_cell.forward(h_prev, x_t)
        H[step + 1] = h_next
        outputs.append(y)

    # Convert the list of outputs to a numpy array
    Y = np.array(outputs)

    return H, Y
