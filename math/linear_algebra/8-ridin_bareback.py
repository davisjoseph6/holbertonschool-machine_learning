#!/usr/bin/env python3

"""
a function that performs matrix multiplication
"""


def mat_mul(mat1, mat2):
    """
    Performs matrix multiplication.

    Parameters:
    - mat1: First 2D matrix (list of lists) of integers or floats.
    - mat2: Second 2D matrix (list of lists) of integers or floats.

    Returns:
    - A new 2D matrix resulting from the matrix mulitplication
    of mat1 and mat2.
    If the two matrices cannot be mulitplied (due to incompatible
    dimensions), return None.
    """
    # Check if the number of columns in mat1 is equal to the
    # number of rows in mat2
    if len(mat1[0]) != len(mat2):
        return None

    # Initialize the resulting matrix with zeros
    result = [[0 for _ in range(len(mat2[0]))] for _ in range(len(mat1))]

    # Perform matrix multiplication
    for i in range(len(mat1)):
        for j in range(len(mat2[0])):
            for k in range(len(mat2)):
                result[i][j] += mat1[i][k] * mat2[k][j]

    return result
