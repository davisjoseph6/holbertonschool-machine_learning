#!/usr/bin/env python3
"""
    Minor
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


def minor(matrix):
    """
    Calculates the minor matrix of a matrix.
    """
    # Validate input matrix
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    for row in matrix:
        if not isinstance(row, list):
            raise TypeError("matrix must be a list of lists")
        if len(row) != len(matrix):
            raise ValueError("matrix must be a non-empty square matrix")

    # Special case: 1x1 matrix
    if len(matrix) == 1:
        return [[1]]

    minor_matrix = []

    for i in range(len(matrix)):
        minor_row = []
        for j in range(len(matrix[i])):
            # Create submatrix removing i-th row and j-th column
            sub_matrix_value = [
                    row[:j] + row[j+1:]
                    for row_idx, row in enumerate(matrix)
                    if row_idx != i
                    ]
            det_sub_matrix = determinant(sub_matrix_value)
            minor_row.append(det_sub_matrix)
        minor_matrix.append(minor_row)

    return minor_matrix
