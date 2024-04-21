#!/usr/bin/env python3
"""
integration functions
"""


def poly_integral(poly, C=0):
    """
    Check validity of poly and C, and handle empty poly
    """
    if (not isinstance(poly, list) or
            not isinstance(C, (int, float)) or
            not poly):
        return None
    if not all(isinstance(x, (int, float)) for x in poly):
        return None

    # Compute the integral
    integral = [C]  # Start with the constant C
    for i in range(len(poly)):
        # Compute each integral part and add to the list
        integral.append(poly[i] / (i + 1))

    # Convert float results to int where applicable
    integral = [int(x) if isinstance(x, float) and x.is_integer()
                else x for x in integral]

    # Remove trailing zeros only if they are not elements after the constant
    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()

    return integral
