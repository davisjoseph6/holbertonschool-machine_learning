#!/usr/bin/env python3
"""
integration functions
"""

def poly_integral(poly, C=0):
    """Calculate the integral of a polynomial.
    
    Args:
        poly (list): List of integers or floats, coefficients of the polynomial.
        C (int, float): The integration constant.
        
    Returns:
        list: Coefficients of the integrated polynomial, or None for invalid inputs.
    """
    # Validate input types
    if not isinstance(poly, list) or not all(isinstance(x, (int, float)) for x in poly):
        return None
    if not isinstance(C, (int, float)):
        return None

    # Handle special case of zero polynomial
    if all(x == 0 for x in poly):
        return [C]

    # Calculate integral coefficients
    integral = [float(C)]
    for i in range(len(poly)):
        new_coef = poly[i] / (i + 1)
        # Append integral coefficients, adjusting for integer values
        integral.append(int(new_coef) if new_coef.is_integer() else new_coef)

    return integral
