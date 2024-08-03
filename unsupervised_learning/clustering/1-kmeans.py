#!/usr/bin/env python3
"""
Initializes cluster centroids for K-means clustering
"""

import numpy as np


def initialize(X, k):
    """
    Initialize cluster centroids for K-means clustering
    """
    if not isinstance(X, np.ndarray) or not isinstance(k, int) or k <= 0:
        return None

    if len(X.shape) != 2:
        return None

    n, d = X.shape
    if n < k:
        return None

    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)

    centroids = np.random.uniform(min_vals, max_vals, (k, d))

    return centroids


def kmeans(X, k, iterations=1000):
    """
    Performs K-means on a dataset.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    n, d = X.shape
    C = initialize(X, k)
    if C is None:
        return None, None

    for i in range(iterations):
        # Calculate distances and assign clusters
        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
        clss = np.argmin(distances, axis=1)

        # Update centroids
        new_C = np.array([X[clss == j].mean(axis=0) if np.any(clss == j) else np.random.uniform(np.min(X, axis=0), np.max(X, axis=0), (1, d)).flatten() for j in range(k)])

        if np.all(C == new_C):
            break
        C = new_C

    return C, clss
