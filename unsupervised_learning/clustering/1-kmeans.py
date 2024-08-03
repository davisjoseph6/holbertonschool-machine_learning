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
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    ctds = initialize(X, k)
    if ctds is None:
        return None, None

    for _ in range(iterations):
        prev_ctds = np.copy(ctds)

        # Calculate distances and assign clusters
        dists = np.sqrt(np.sum((X - ctds[:, np.newaxis]) ** 2, axis=2))
        clss = np.argmin(dists, axis=0)

        for i in range(k):
            # Mask: points present in cluster
            cluster_mask = X[clss == i]
            if len(cluster_mask) == 0:
                ctds[i] = initialize(X, 1)
            else:
                ctds[i] = np.mean(cluster_mask, axis=0)

        # Recalculate distances and reassign clusters
        dists = np.sqrt(np.sum((X - ctds[:, np.newaxis]) ** 2, axis=2))
        clss = np.argmin(dists, axis=0)

        # Convergence check (if points haven't changed clusters)
        if np.allclose(ctds, prev_ctds):
            break

    return ctds, clss
