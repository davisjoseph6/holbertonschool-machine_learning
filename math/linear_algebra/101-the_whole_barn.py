#!/usr/bin/env python3
"""
Module to add n dimension matrices with the same shape
"""


def add_matrices(mat1, mat2):
    """
    Adds two matrices of the same shape.
    Args:
        mat1, mat2: Given matrices.
    Returns:
        list: The sum of the two matrices, or None if their shapes differ
    """
    if isinstance(mat1, list) and isinstance(mat2, list):
        if len(mat1) != len(mat2):
            return None
        mat1_lists = any(isinstance(sub, list) for sub in mat1)
        mat2_lists = any(isinstance(sub, list) for sub in mat2)
        if mat1_lists != mat2_lists:
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
