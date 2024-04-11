#!/usr/bin/env python3
"""
A function that adds two matrices
"""


def get_shape(matrix):
    """Utility function to get the shape of a nested matrix.
    """
    if not isinstance(matrix, list):
        return []
    if not matrix:
        return [0]
def add_matrices(mat1, mat2):
    """
    Adds two matrices element-wise.

    Returns:
    - A new matrix containing the element-wise sums of mat1 and mat2.
    Returns None if the matrices have different shapes.
    """
    shape1 = get_shape(mat1)
    shape2 = get_shape(mat2)
    if shape1 != shape2:
        return None  # Shape mismatch

    if not isinstance(mat1, list):
        return mat1 + mat2  # Base case

    return [add_matrices(sub_mat1, sub_mat2) for sub_mat1, sub_mat2 in zip(mat1, mat2)]
