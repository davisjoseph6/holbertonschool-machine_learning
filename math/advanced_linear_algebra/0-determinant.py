#!/usr/bin/env python3
"""
A function that calculates the determinant of a matrix
"""


def sub_matrix(matrix, i):
    """
    Creates a submatrix by removing the first row and the i-th column
    """
    if not matrix:
        return []

    matrix2 = []
    for row in matrix[1:]:  # Skip the first row
        matrix2.append(row[:i] + row[i + 1:])  # Remove the i-th column

    return matrix2


def determinant(matrix):
    """
    Calculates the determinant of a matrix.
    """
    # test if matrix is a list of lists
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    for sub_list in matrix:
        if not isinstance(sub_list, list):
            raise TypeError("matrix must be a list of lists")

    # Special case: empty matrix (0x0)
    if len(matrix[0]) == 0:
        return 1

    # test if matrix is square
    if len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a square matrix")

    # Base cases for small matrices
    if len(matrix) == 1:
        return matrix[0][0]
    elif len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[1][0] * matrix[0][1]

    # Recursive case: Calculate the determinant using cofactor expansion
    det = 0
    for i in range(len(matrix[0])):
        det += (
                ((-1) ** i) * matrix[0][i] * determinant(
                    sub_matrix(matrix, i)
                    )
                )
    return det
