#!/usr/bin/env python3
"""
integration functions
"""


def poly_integral(poly, C=0):
    """Calculate the integral of a polynomial represented by coefficients."""
    # Validate input types and structure
    if not isinstance(poly, list) or not all(isinstance(x, (int, float)) for x in poly):
        return None
    if not isinstance(C, (int, float)):
        return None
    
    # Check for polynomial content or C significance
    if not poly or all(x == 0 for x in poly):
        return None  # Adjust this line to return None instead of handling zero polynomial as valid

    # Calculate integral
    integral = [C] + [poly[i] / (i + 1) for i in range(len(poly))]
    # Convert to integers where appropriate
    integral = [int(x) if isinstance(x, float) and x.is_integer() else x for x in integral]
    # Remove unnecessary trailing zeros
    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()
    return integral

