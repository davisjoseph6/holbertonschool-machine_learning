#!/usr/bin/env python3
"""
Conducts forward propagation using Dropout.
"""

import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Conducts forward propagation using Dropout.
    """
    cache = {}
    cache['A0'] = X

    for layer in range(1, L + 1):
        W = weights['W' + str(layer)]
        b = weights['b' + str(layer)]
        A_prev = cache['A' + str(layer - 1)]

        Z = np.matmul(W, A_prev) + b

        if layer != L:
            A = np.tanh(Z)
            D = np.random.rand(A.shape[0], A.shape[1]) < keep_prob
            A = A * D
            A = A / keep_prob
            cache['D' + str(layer)] = D.astype(int)
        else:
            exp_Z = np.exp(Z)
            A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

        cache['A' + str(layer)] = A

    return cache
