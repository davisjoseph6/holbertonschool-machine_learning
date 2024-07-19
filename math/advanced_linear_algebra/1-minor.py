#!/usr/bin/env python3
"""
a function that calculates the minor matrix of a matrix
"""


def sub_matrix(matrix, row, col):
    """
    Creates a submatrix by removing the specified row and column.
    """
    return [r[:col] + r[col + 1:] for r in (matrix[:row] + matrix[row + 1:])]


def determinant(matrix):
    """
    Calculates the determinant of a matrix.
    """
    # Base cases for small matrices
    if len(matrix) == 1:
        return matrix[0][0]
    elif len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[1][0] * matrix[0][1]

    # Recursive case: Calculate the determinant using cofactor expansion
    det = 0
    for c in range(len(matrix)):
        det += ((-1) ** c) * matrix[0][c] * determinant(sub_matrix(matrix, 0, c))
    return det


def minor(matrix):
    """
    Calculates the minor matrix of a matrix.
    """
    # Validate input matrix
    if not isinstance(matrix, list):
        raise ValueError("matrix must be a list of lists")
    if len(matrix) == 0 or any(not isinstance(row, list) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")
    if any(len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    # Handle the 1x1 case
    if len(matrix) == 1:
        return [[1]]

    # Compute the minor matrix
    size = len(matrix)
    minor_matrix = []
    for i in range(size):
        minor_row = []
        for j in range(size):
            sub_m = sub_matrix(matrix, i, j)
            minor_row.append(determinant(sub_m))
        minor_matrix.append(minor_row)

    return minor_matrix
