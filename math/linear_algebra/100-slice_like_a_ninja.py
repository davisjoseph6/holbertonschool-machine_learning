#!/usr/bin/env python3
"""
A function that slices a matrix along specific axes
"""


def np_slice(matrix, axes={}):
    """
    Slices a matrix along specific axes.

    Parameters:
    - matrix: A numpy.ndarray to be sliced
    - axes: A dictionary where the key is an axis to slice along,
    and the value is a tuple representing the slice to make along
    that axis.

    Returns:
    - A new numpy.ndarray after applying the slicing
    """
    # Create a tuple of slices for each axis
    slices = [
            slice(*axes.get(axis, (None, None))) for axis in range(matrix.ndim)
            ]

    # Use the tuple of slices to slice the matrix
    return matrix[tuple(slices)]
