#!/usr/bin/env python3
"""
A function that calculates the determinant of a matrix
"""


def sub_matrix(matrix, i):
    """
        function to create submatrix 2x2

        :param matrix: initial matrix
        :param i: row, column to pop

        :return: submatrix 2x2
    """
    if not matrix:
        return []

    matrix2 = []
    for row in matrix[1:]:
        matrix2.append(row[:i] + row[i + 1:])

    return matrix2


def determinant(matrix):
    """
        function that calculates the determinant of a matrix

        :param matrix: list of lists whose determinant should be calculated

        :return: determinant of matrix
    """
    # Test list of list format
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    for sub_list in matrix:
        if not isinstance(sub_list, list):
            raise TypeError("matrix must be a list of lists")

    d = 0

    # special case
    if len(matrix[0]) == 0:
        return 1

    # test square format of matrix
    if len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a square matrix")

    # special case only one element
    if len(matrix) == 1:
        d = matrix[0][0]
        return d
    # simple matrix of 2x2
    elif len(matrix) == 2:
        d = matrix[0][0] * matrix[1][1] - matrix[1][0] * matrix[0][1]
        return d
    # recursiv action to sum all sub_matrix
    else:
        for i in range(len(matrix[0])):
            d += (((-1) ** i) * matrix[0][i] *
                  determinant(sub_matrix(matrix, i)))
        return d
