#!/usr/bin/env python3
"""
a function that calculates the mean and covariance of a data set.
"""

import numpy as np


def mean_cov(X):
    """
    a function that calculates the mean and covariance of a data set.
    """
    # Check if X is a 2D numpy array
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise TypeError("X must be a 2D numpy.ndarray")

    # Get the number of data points (n) and the number of dimensions (d)
    n, d = X.shape

    # Check if there are at least 2 data points
    if n < 2:
        raise ValueError("X must contain multiple data points")

    # Calculate the mean
    mean = np.mean(X, axis=0, keepdims=True)

    # Calculate the covariance matrix
    X_centered = X - mean
    cov = np.dot(X_centered.T, X_centered) / (n - 1)

    return mean, cov
