#!/usr/bin/env python3
"""
    Cofactor
"""

import numpy as np


def definiteness(matrix):
    """
    Calculates the definiteness of a matrix.
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return None

    try:
        eigenvalues = np.linalg.eigvals(matrix)
    except np.linalg.LinAlgError:
        return None

    positive_count = np.sum(eigenvalues > 0)
    negative_count = np.sum(eigenvalues < 0)
    zero_count = np.sum(eigenvalues == 0)

    if positive_count == len(eigenvalues):
        return "Positive definite"
    if positive_count > 0 and zero_count > 0 and negative_count == 0:
        return "Positive semi-definite"
    if negative_count == len(eigenvalues):
        return "Negative definite"
    if negative_count > 0 and zero_count > 0 and positive_count == 0:
        return "Negative semi-definite"
    if positive_count > 0 and negative_count > 0:
        return "Indefinite"

    return None
