#!/usr/bin/env python3
"""
A function that calculates the integral of a polynomial
"""


def poly_integral(poly, C=0):
    """
    Validate input
    """
    if not isinstance(poly, list) or not all(isinstance(coef, (int, float)) for coef in poly):
        return None
    if not isinstance(C, int):
        return None

    if all(coef == 0 for coef in poly):
        return [C]  # Return [C] if the polynomial is zero

    integral = [C]  # Start with the integration constant
    for i, coef in enumerate(poly):
        new_coef = coef / (i + 1)  # Calculate the new coefficient
        integral.append(int(new_coef) if new_coef.is_integer() else new_coef)  # Append as int if whole number, else as float

    # remove trailing zeros if they are not the only coefficient
    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()

    return integral
