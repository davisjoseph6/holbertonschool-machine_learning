#!/usr/bin/env python3
"""
    Minor
"""
import numpy as np


def definiteness(matrix):
    """
        function that calculates the definiteness of a matrix

        :param matrix: ndarray, shape(n,n)

        :return: - Positive definite
                 - Positive semi-definite
                 - Negative semi-definite
                 - Negative definite
                 - Indefinite
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")
    if matrix.size == 0:
        return None
    # test square matrix
    if matrix.shape[0] != matrix.shape[1]:
        return None
    # check matrix symetric
    if not np.array_equal(matrix, matrix.T):
        return None

    eigen_val, _ = np.linalg.eig(matrix)

    # determin min and max eigen values
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
