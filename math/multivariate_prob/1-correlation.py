#!/usr/bin/env python3
"""
a function that calculates a correlation matrix
"""

import numpy as np


def correlation(C):
    """
    a function that calculates a correlation matrix
    """
    # Check if C is a numpy array
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")

    # Check if C is a square matrix
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    # Calculate the standard deviations
    std_devs = np.sqrt(np.diag(C))

    # Create the outer product of the standard deviations
    std_devs_outer = np.outer(std_devs, std_devs)

    # Calculate the correlation matrix
    correlation_matrix = C / std_devs_outer

    return correlation_matrix
