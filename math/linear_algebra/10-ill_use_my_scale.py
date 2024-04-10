#!/usr/bin/env python3
"""
a function that calculates the shape of a numpy.ndarray
"""


def infer_shape(matrix):
    """
    Infer the shape of a list structure that represents a matrix
    or a higher dimensional array
    """
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0] if matrix else []
    return tuple(shape)
