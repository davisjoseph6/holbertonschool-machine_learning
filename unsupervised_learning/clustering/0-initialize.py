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
