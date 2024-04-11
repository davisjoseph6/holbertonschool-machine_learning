#!/usr/bin/env python3
"""
A function that performs element-wise arithmetic
operations on numpy.ndarrays.
"""


def np_elementwise(mat1, mat2):
    """
    Performs element-wise addition, subtraction, multiplication,
    and division
    """
    addition = mat1 + mat2
    subtraction = mat1 - mat2
    multiplication = mat1 * mat2
    division = mat1 / mat2
    return (addition, subtraction, multiplication, division)
