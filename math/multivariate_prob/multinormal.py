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

    def pdf(self, x):
        """
        Calculate the probability density function (PDF) value for a given
        data point.
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")

        d = self.mean.shape[0]

        if x.shape != (d, 1):
            raise ValueError(f"x must have the shape ({d}, 1)")

        x_centered = x - self.mean
        inv_cov = np.linalg.inv(self.cov)
        det_cov = np.linalg.det(self.cov)

        norm_factor = 1 / np.sqrt((2 * np.pi) ** d * det_cov)
        exp_factor = np.exp(
                -0.5 * np.dot(np.dot(x_centered.T, inv_cov), x_centered))

        return float(norm_factor * exp_factor)
