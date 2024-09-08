#!/usr/bin/env python3
""" Deep RNN forward propagation """

import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    Performs forward propagation for a deep RNN.
    """
    t, m, i = X.shape
    l, _, h = h_0.shape

    # Initialize containers for hidden states and outputs
    H = np.zeros((t + 1, l, m, h))
    H[0] = h_0
    Y = []

    # Forward propagation through time and layers
    for step in range(t):
        x_t = X[step]  # Extract the input at the current time step

        # Propagate through each layer
        for layer in range(l):
            rnn_cell = rnn_cells[layer]
            h_prev = H[step, layer]
            h_next, y = rnn_cell.forward(h_prev, x_t)
            H[step + 1, layer] = h_next

            # The output of this layer becomes the input to the next layer
            x_t = h_next

        # Append the output of the final layer
        Y.append(y)

    Y = np.array(Y)  # Convert the list of outputs to a numpy array
    return H, Y
