#!/usr/bin/env python3
"""
Module to add n dimension matrices with the same shape
"""


def add_matrices(mat1, mat2):
    """
    Add n-dimensional matrices with the same shape.
    Args:
        mat1, mat2: Given matrices.
    Returns:
        The recursively computed addition of mat1 and mat2,
        or None if the matrices have different shapes
    """
    if isinstance(mat1, list) and isinstance(mat2, list):
        if len(mat1) != len(mat2) or any(isinstance(sub, list) for sub in mat1) != any(isinstance(sub, list) for sub in mat2):
            return None

        result = []
        for i in range(len(mat1)):
            added = add_matrices(mat1[i], mat2[i])
            if added is None:
                return None
            result.append(added)
        return result
    else:
        return mat1 + mat2
