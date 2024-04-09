#!/usr/bin/env python3
"""
A function that returns te transpose of a 2D matrix
"""


def matrix_transpose(matrix):
    """
    Transpose a 2D matrix.
    """
    return [list(row) for row in zip(*matrix)]
