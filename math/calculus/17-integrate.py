#!/usr/bin/env python3
"""
A module for integrating polynomials.
"""

def poly_integral(poly, C=0):
    """
    Calculate the integral of a polynomial represented by a list of coefficients.

    Args:
    poly (list of int, float): Coefficients of the polynomial where the index
                                represents the power of x.
    C (int): An integration constant.

    Returns:
    list: A new list of coefficients after integration or None if input is invalid.
    """
    # Validate input types
    if not isinstance(poly, list) or not all(isinstance(x, (int, float)) for x in poly):
        return None
    if not isinstance(C, int):
        return None

    # Early return for a polynomial which is effectively zero
    if not poly or all(x == 0 for x in poly):
        return [C] if C != 0 else [0]

    # Compute the integral coefficients
    integral = [float(C)]  # Start the new list with the integration constant
    for i in range(len(poly)):
        new_coef = poly[i] / (i + 1)
        # Check if the new coefficient is an integer
        if new_coef.is_integer():
            new_coef = int(new_coef)
        integral.append(new_coef)

    # Remove trailing zeros to minimize the list size
    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()

    return integral

