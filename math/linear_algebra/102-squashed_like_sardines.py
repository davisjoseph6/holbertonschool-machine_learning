#!/usr/bin/env python3
"""
A function that concatenates two matrices alonga specific axis
"""


def cat_matrices(mat1, mat2, axis=0):
    """
    Concatenates two matrcies along a specified axis.

    Returns:
    -list: New matrix resulting from concatenation of mat1 and mat2
    - None: If matrices cannot be concatenated due to incompatible dimensions
    """

    # Helper funtion to get the shape of the matrix
    def get_shape(matrix):
        if not isinstance(matrix, list) or not matrix:
            return []
        return [len(matrix)] + get_shape(matrix[0])

    # Get shapes of both matrices
    shape1 = get_shape(mat1)
    shape2 = get_shape(mat2)

    # Check if the matrices can be concatenated along the specified axis
    if len(shape1) != len(shape2) or any(s1 != s2 for s1, s2 in zip(shape1[:axis], shape2[:axis])) \
            or any(s1 != s2 for s1, s2 in zip(shape1[axis+1:], shape2[axis+1:])):
                return None

    # Recursiveeeeeeeee concatenation of matrices
    if axis == 0:
        return mat1 + mat2
    else:
        return [cat_matrices(m1, m2, axis-1) for m1, m2 in zip(mat1, mat2)]

