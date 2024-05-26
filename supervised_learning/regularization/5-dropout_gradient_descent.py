#!/usr/bin/env python3
"""
Updates the weights of a neural network with Dropout regularization using
gradient descent.
"""

import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Updates the weights of a neural network with Dropout regularization using
    gradient descent.
    """
    m = Y.shape[1]
    dZ = cache['A' + str(L)] - Y

    for layer in reversed(range(1, L + 1)):
        A_prev = cache['A' + str(layer - 1)]
        W = weights['W' + str(layer)]
        b = weights['b' + str(layer)]

        dW = (1 / m) * np.matmul(dZ, A_prev.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        if layer > 1:
            D = cache['D' + str(layer - 1)]
            dA_prev = np.matmul(W.T, dZ)
            dA_prev = dA_prev * D
            dA_prev = dA_prev / keep_prob
            dZ = dA_prev * (1 - np.square(A_prev))

        weights['W' + str(layer)] = W - alpha * dW
        weights['b' + str(layer)] = b - alpha * db
