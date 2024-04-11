#!/usr/bin/env python3
"""
that adds two arrays element-wise
"""


def add_arrays(arr1, arr2):
    """
    Add two arrays element-wise

    Parameters:
    -   arr1: First list of integers or floats.
    -   arr2: Second list of integers or floats.
    """
    if len(arr1) != len(arr2):
        return None

    return [a + b for a, b in zip(arr1, arr2)]
