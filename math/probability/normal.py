#!/usr/bin/env python3
"""
This module contains the Normal class for representing a normal distribution.
"""


class Normal:
    """
    Represents a normal distribution.
    """
    def __init__(self, data=None, mean=0., stddev=1.):
        """
        Initializes the Normal distribution with data or given mean and stddev.
        """
        if data is None:
            if stddev <= 0:
                raise ValueError('stddev must be a positive value')
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            self.mean = float(sum(data) / len(data))
            variance = sum((x - self.mean) ** 2 for x in data) / len(data)
            self.stddev = float(variance ** 0.5)

    def z_score(self, x):
        """
        Calculates the z-score of a given x-value.
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """
        Calculates the x-value of a given z-score.
        """
        return z * self.stddev + self.mean

    def pdf(self, x):
        """
        Calculates the value of the PDF for a given x-value.
        """
        pi = 3.1415926536
        e = 2.7182818285
        coef = 1 / (self.stddev * (2 * pi) ** 0.5)
        exponent = -0.5 * ((x - self.mean) / self.stddev) ** 2
        return coef * e ** exponent

    def cdf(self, x):
        """
        Calculates the value of the CDF for a given x-value.
        """
        pi = 3.1415926536
        z = (x - self.mean) / (self.stddev * (2 ** 0.5))
        erf = 1.128379167 * (z - (z ** 3) / 3 + (z ** 5) / 10 -
                             (z ** 7) / 42 + (z ** 9) / 216)
        return 0.5 * (1 + erf)
