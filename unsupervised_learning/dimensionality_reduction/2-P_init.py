#!/usr/bin/env python3
"""
t-SNE P affinities initialization
"""
import numpy as np


def P_init(X, perplexity):
    """
    Initializes all variables required to calculate the P affinities in t-SNE.
    """
    n, d = X.shape

    # Compute the squared pairwise distance matrix D
    sum_X = np.sum(np.square(X), axis=1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    np.fill_diagonal(D, 0)

    # Initialize P, betas, and H
    P = np.zeros((n, n))
    betas = np.ones((n, 1))
    H = np.log2(perplexity)

    return D, P, betas, H
