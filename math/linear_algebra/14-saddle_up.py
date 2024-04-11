#!/usr/bin/env python3
"""
A function that performs matrix multiplication
"""

import numpy as np


def np_matmul(mat1, mat2):
    """
    Performs matrix multiplication between two numpy.ndarrays.

    Returns:
    - A new numpy.ndarray resulting from the matrix multiplication of mat1
    and mat2.
    """
    return mat1 @ mat2
