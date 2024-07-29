#!/usr/bin/env python3
"""
t-SNE gradients calculation
"""
import numpy as np
Q_affinities = __import__('5-Q_affinities').Q_affinities


def grads(Y, P):
    """
    Calculates the gradients of Y
    """
    Q, num = Q_affinities(Y)
    PQ_diff = P - Q
    dY = np.zeros_like(Y)
    n, ndim = Y.shape

    for i in range(n):
        dY[i] = np.sum(np.expand_dims(PQ_diff[:, i] * num[:, i],
                                      axis=1) * (Y[i] - Y), axis=0)

    return dY, Q
