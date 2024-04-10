#!/usr/bin/env python3
"""
a function that adds two matrices element-wise
"""


def add_matrices2D(mat1, mat2):
    """
    Adds two 2D matrices element-wise.

    Parameters:
    - mat1: First 2D matrix (list of lists) of integers or floats
    -mat2: Second 2D matrix (list of lists) of integers or floats.

    Returns:
    - A new 2D matrix containing the element-wise sums of mat1 and mat2 if they are of the same shape, otherwise None.
    """
    # Check if the two matrices have the same shape
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None

    # Add the matrices element-wise
    return [[mat1[i][j] + mat2[i][j] for j in range(len(mat1[0]))] for i in range(len(mat1))]
