#!/usr/bin/env python3
"""
A function that calculates the integral of a polynomial
"""


def poly_integral(poly, C=0):
    # Validate input
    if not isinstance(poly, list) or not all(isinstance(coef, (int, float)) for coef in poly):
        return None
    if not isinstance(C, int):
        return None

    # Check if poly is empty
    if not poly:
        return [C]

    # Calculate the integral
    integral = [C]  # Start with the integration constant
    for index, coef in enumerate(poly):
        if coef == 0:
            new_coef = 0
        else:
            new_power = index + 1
            new_coef = coef / new_power

        # Ensure to handle new_coef as float to apply is_integer()
        if isinstance(new_coef, float) and new_coef.is_integer():
            new_coef = int(new_coef)

        integral.append(new_coef)

    # Remove trailing zeros to make the list as small as possible
    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()

    return integral
