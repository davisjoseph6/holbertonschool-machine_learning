#!/usr/bin/env python3
"""
This module contains the Exponential class for representing an exponential
distribution.
"""


class Exponential:
    """
    Represents an exponential distribution.
    """
    def __init__(self, data=None, lambtha=1.):
        """
        Initializes the Exponential distribution with data or lambtha.
        """
        if data is None:
            if lambtha <= 0:
                raise ValueError('lambtha must be a positive value')
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            self.lambtha = 1 / (sum(data) / len(data))
