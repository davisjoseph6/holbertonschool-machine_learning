#!/usr/bin/env python3
"""
A class that represents a multivariate normal distribution.
"""

import numpy as np


class MultiNormal:
    """
    A class that represents a multivariate normal distribution.
    """
    def __init__(self, data):
        """
        multivariate normal distribution.
        """
        # check if data is a 2D numpy array
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        # Get the number of dimensions (d) and the number of data points (n)
        d, n = data.shape

        # Check if there are at least 2 data points
        if n < 2:
            raise ValueError("data must contain multiple data points")

        # Calculate the mean
        self.mean = np.mean(data, axis=1, keepdims=True)

        # Calculate the covariance matrix
        data_centered = data - self.mean
        self.cov = np.dot(data_centered, data_centered.T) / (n - 1)
