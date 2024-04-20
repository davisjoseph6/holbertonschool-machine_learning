#!/usr/bin/env python3
"""
A function that calculates the integral of a polynomial
"""


def poly_integral(poly, C=0):
    """
    Validate input
    """
    if not isinstance(poly, list) or not all(isinstance(x, (int, float)) for x in poly):
        return None

    if not isinstance(C, (int, float)):
        return None

    if not poly or all(x == 0 for x in poly):
        return [C] if C != 0 else [0]

    arr = [C] + [poly[i] / (i + 1) for i in range(len(poly))]

    return [int(x) if isinstance(x, float) and x.is_integer() else x for x in arr]
