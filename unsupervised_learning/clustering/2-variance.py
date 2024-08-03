#!/usr/bin/env python3
"""
Calculates the total intra-cluster variance for a dataset.
"""

import numpy as np


def variance(X, C):
    """
    Caluclates the total intra-cluster variance for a dataset.
    """
    if not isinstance(X, np.ndarray) or not isinstance(C, np.ndarray):
        return None
    if X.ndim != 2 or C.ndim != 2:
        return None
    if X.shape[1] != C.shape[1]:
        return None

    # Calculate the distances from each point to each centroid
    distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)

    # Assign each point to the closest centroid
    min_distances = np.min(distances, axis=1)

    # Calculate the variance
    var = np.sum(min_distances ** 2)

    return var
