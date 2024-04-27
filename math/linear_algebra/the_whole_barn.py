#!/usr/bin/env python3
"""
A function that adds two matrices.
"""


def add_matrices(mat1, mat2):
    """
    Adds two matrices element-wise.

    Returns:
    - A new matrix containing the element-wise sums of mat1 and mat2.
    Returns None if the matrices have different shapes.
    """
    if type(mat1) != list or type(mat2) != list:
        return mat1 + mat2  # Base case: both are numbers, directly add them

    if len(mat1) != len(mat2):
        return None  # Different shapes, cannot add

    # Check if we are dealing with matrices of higher dimensions
    if isinstance(mat1[0], list):
        if any(len(submat1) != len(submat2)
                for submat1, submat2 in zip(mat1, mat2)):
            return None  # Sub-matrices have different shapes

        # Recursive step: add sub-matrices
        return [add_matrices(submat1, submat2)
                for submat1, submat2 in zip(mat1, mat2)]
    else:
        # Simple case: both mat1 and mat2 are 1D, directly add
        # corresponding elements
        return [item1 + item2 for item1, item2 in zip(mat1, mat2)]
