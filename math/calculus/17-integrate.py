#!/usr/bin/env python3
"""
integration functions
"""


def poly_integral(poly, C=0):
    """Calculate the integral of a polynomial represented by coefficients."""
    if isinstance(poly, list) and all(isinstance(x, (int, float)) for x in poly) and isinstance(C, (int, float)):
        if not poly:  # Handle empty polynomial list explicitly
            return [C] if C != 0 else [0]
        integral = [C] + [poly[i] / (i + 1) for i in range(len(poly))]
        # Convert to integers where possible
        integral = [int(x) if isinstance(x, float) and x.is_integer() else x for x in integral]
        # Remove trailing zeros from the integral
        while len(integral) > 1 and integral[-1] == 0:
            integral.pop()
        return integral
    return None
