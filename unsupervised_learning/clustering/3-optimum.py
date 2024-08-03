#!/usr/bin/env python3
"""
Determines the optimum number of clusters by variance for a dataset.
"""

import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, lmax=None, iterations=1000):
    """
    Determines the optimum number of clusters by variance for a dataset
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None
    if not isinstance(kmin, int) or kmin <= 0:
        return None, None
    if kmax is None:
        kmax = X.shape[0]
    if not isinstance(kmax, int) or kmax <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None
    if kmax < kmin:
        return None, None

    results = []
    d_vars = []

    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        if C is None or clss is None:
            return None, None
        results.append((C, clss))
        var = variance(X, C)
        if var is None:
            return None, None
        if k == kmin:
            initial_var = var
        d_vars.append(initial_var - var)

    return results, d_vars
