#!/usr/bin/env python3
"""
t-SNE Q affinities calculation
"""
import numpy as np


def Q_affinities(Y):
    """
    Calculates the Q affiniies
    """
    n, ndim = Y.shape

    # Compute the squared pairwise distance matrix
    sum_Y = np.sum(np.square(Y), axis=1)
    D = np.add(np.add(-2 * np.dot(Y, Y.T), sum_Y).T, sum_Y)

    # Compute the numerator of the Q affinities
    num = 1 / (1 + D)
    np.fill_diagonal(num, 0)

    # Cimpute the Q affinities
    Q = num / np.sum(num)

    return Q, num
