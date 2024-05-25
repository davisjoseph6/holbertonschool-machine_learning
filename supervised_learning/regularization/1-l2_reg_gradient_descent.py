#!/usr/bin/env python3
"""
Updates the weights and biases of a neural network using gradient
descent with L2 regularization.
"""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates the wieghts and biases
    """
    m = Y.shape[1]
    A_prev = cache['A' + str(L - 1)]
    A_L = cache['A' + str(L)]

    dZ = A_L - Y

    for l in range(L, 0, -1):
        A_prev = cache['A' + str(l - 1)]
        W = weights['W' + str(l)]
        b = weights['b' + str(l)]

        dW = (1 / m) * np.matmul(dZ, A_prev.T) + (lambtha / m) * W
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        weights['W' + str(l)] = W - alpha * dW
        weights['b' + str(l)] = b - alpha * db

        if l > 1:
            dA_prev = np.matmul(W.T, dZ)
            dZ = dA_prev * (1 - A_prev ** 2)
