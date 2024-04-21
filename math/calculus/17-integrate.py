#!/usr/bin/env python3
"""
integration functions
"""


def poly_integral(poly, C=0):
    # Extend the check to handle cases where poly might be empty in a way that should return None
    if not isinstance(poly, list) or not isinstance(C, (int, float)) or not poly:
        return None
    if not all(isinstance(x, (int, float)) for x in poly):
        return None

    integral = [C] + [poly[i] / (i + 1) for i in range(len(poly))]
    integral = [int(x) if isinstance(x, float) and x.is_integer() else x for x in integral]
    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()
    return integral

