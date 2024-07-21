#!/usr/bin/env python3
"""
This module contains the Binomial class for representing a binomial
distribution.
"""


class Binomial:
    """
    Represents a binomial distribution.
    """
    def __init__(self, data=None, n=1, p=0.5):
        """
        Initializes the Binomial distribution with data or given n and p.
        """
        if data is None:
            if n <= 0:
                raise ValueError('n must be a positive value')
            if p <= 0 or p >= 1:
                raise ValueError('p must be greater than 0 and less than 1')
            self.n = int(n)
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            mean = sum(data) / len(data)
            variance = sum((x - mean) ** 2 for x in data) / len(data)
            self.p = 1 - variance / mean
            self.n = round(mean / self.p)
            self.p = mean / self.n

    def pmf(self, k):
        """
        Calculates the value of the PMF for a given number of successes.
        """
        from math import factorial

        k = int(k)
        if k < 0 or k > self.n:
            return 0

        nCk = factorial(self.n) / (factorial(k) * factorial(self.n - k))
        return nCk * (self.p ** k) * ((1 - self.p) ** (self.n - k))
