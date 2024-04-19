#!/usr/bin/env python3
"""
Calculate the sum of squares from 1 to n using the formula:
    Sum = n(n+1)(2n+1)/6
"""


def summation_i_squared(n):
    """
    Parameters:
    n (int): The number up to which the square of numbers
    are summed
    
    Returns:
    int: The sum of squares from 1 to n
    None: If n is not a valid integer
    """
    if not isinstance(n, int) or n < 1:  # Validate that n is a positive integer
        return None
    return (n * (n +1) * (2 * n + 1)) // 6  # Calculate and return the sum 
