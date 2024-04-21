#!/usr/bin/env python3
"""
integration functions
"""


def poly_integral(poly, C=0):
    if not isinstance(poly, list) or not all(isinstance(x, (int, float)) for x in poly) or not isinstance(C, (int, float)):
        return None

    # Calculate the integral only if there are coefficients
    if poly:
        integral = [C] + [poly[i] / (i + 1) for i in range(len(poly))]
        # Remove unnecessary zeros from the result
        integral = [int(x) if isinstance(x, float) and x.is_integer() else x for x in integral]
        # Remove trailing zeros only if they are not the only elements after the constant
        while len(integral) > 1 and integral[-1] == 0:
            integral.pop()
        return integral
    else:
        # If the polynomial is empty, return just the constant
        return [C]

