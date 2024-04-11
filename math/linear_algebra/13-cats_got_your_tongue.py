#!/usr/bin/env python3
"""
A function that concatenates two matrices along a specific axis
"""

import numpy as np


def np_cat(mat1, mat2, axis=0):
    """
    Parameters:
    - mat1: A numpy.ndarray
    - mat2: A numpy.ndarray
    - axis: The axis along which to concatenate the matrices.

    Returns:
    - A new numpy.ndarray resulting from concate,ato,g mat1 and
    mat2 along the specific axis
    """
    return np.concatenate((mat1, mat2), axis=axis)
