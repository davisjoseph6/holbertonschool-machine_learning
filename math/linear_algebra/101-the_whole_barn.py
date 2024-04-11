#!/usr/bin/env python3
"""
A function to add n dimension matrices with the same shape
"""


def shape(matrix):
    """
    Return the shape of a matrix.
    Args:
        matrix: Given matrix.
    Returns:
        The shape of the matrix as a list of integers.
    """
    if type(matrix[0]) != list:
        return [len(matrix)]
    else:
        return [len(matrix)] + shape(matrix[0])

def rec_matrix(mat1, mat2):
    """
    Recursively operate an add of a n-dimensional matrix.
    Args:
        mat1, mat2: Given matrices.
    Returns:
        The addition of mat1 and mat2 iterating recursively.
    """
    new_mat = []

    if (type(mat1) and type(mat2)) == list:
        for i in range(len(mat1)):
            if type(mat1[i]) == list:
                new_mat.append(rec_matrix(mat1[i], mat2[i]))
            else:
                new_mat.append(mat1[i] + mat2[i])
        return new_mat

def add_matrices(mat1, mat2):
    """
    Add n-dimensional matrices with the same shape.
    Args:
        mat1, mat2: Given matrices.
    Returns:
        The recursively computed addition of mat1 and mat2,
        or None if the matrices have different shapes
    """
    if shape(mat1) != shape(mat2):
        return None
    else:
        new_mat = rec_matrix(mat1, mat2)
        return new_mat
