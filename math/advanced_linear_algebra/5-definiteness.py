#!/usr/bin/env python3
"""
Definiteness
"""

import numpy as np


def definiteness(matrix):
    """
    Calculates the definiteness of a matrix.
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")
    if matrix.size == 0:
        return None
    if matrix.shape[0] != matrix.shape[1]:
        return None
    if not np.array_equal(matrix, matrix.T):
        return None

    eigen_val, _ = np.linalg.eig(matrix)

    min_eig = np.min(eigen_val)
    max_eig = np.max(eigen_val)

    if min_eig > 0 and max_eig > 0:
        return "Positive definite"
    elif min_eig == 0 and max_eig > 0:
        return "Positive semi-definite"
    elif min_eig < 0 and max_eig < 0:
        return "Negative definite"
    elif min_eig < 0 and max_eig == 0:
        return "Negative semi-definite"
    elif min_eig < 0 < max_eig:
        return "Indefinite"
    else:
        return None
