#!/usr/bin/env python3
"""
Initializes variables for a Gaussian Mixture Model.
"""

import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """
    Initializes variables for a Gaussian Mixture Model.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None

    n, d = X.shape

    # Initialize pi evenly
    pi = np.full((k,), 1 / k)

    # Initialize centroids using K-means
    m, _ = kmeans(X, k)
    if m is None:
        return None, None, None

    # Initialize covariance matrices as identity matrices
    S = np.tile(np.identity(d), (k, 1, 1))

    return pi, m, S
