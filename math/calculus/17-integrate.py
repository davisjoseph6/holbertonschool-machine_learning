#!/usr/bin/env python3
"""
A function that calculates the integral of a polynomial
"""

def poly_integral(poly, C=0):
    """
    Calculate the integral of a polynomial represented by its coefficients.

    Args:
    poly (list of int/float): List of coefficients, where the index represents the power of x.
    C (int, optional): Integration constant, default is 0.

    Returns:
    list of float/int: Integral of the polynomial as coefficients of the resulting polynomial.
    None: If input is invalid.
    """

    # Validate inputs
    if not isinstance(poly, list) or not all(isinstance(x, (int, float)) for x in poly):
        return None
    if not isinstance(C, (int, float)):
        return None

    # Handle edge case of an empty list or all zeros
    if not poly or all(x == 0 for x in poly):
        return [C] if C != 0 else [0]

    # Calculate integral coefficients
    arr = [C] + [poly[i] / (i + 1) for i in range(len(poly))]

    # Convert floats to integers if they represent whole numbers
    return [int(x) if isinstance(x, float) and x.is_integer() else x for x in arr]
