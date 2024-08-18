#!/usr/bin/env python3
"""
Module documentation for '3-bayes_opt'.
"""

import numpy as np
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """
    Class that performs Bayesian optimization on a
    noiseless 1D Gaussian process.
    """

    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1, sigma_f=1,
                 xsi=0.01, minimize=True):
        """
        Class constructor.
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l=l, sigma_f=sigma_f)

        # Generate ac_samples evenly spaced points within the bounds
        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples).reshape(-1, 1)

        self.xsi = xsi
        self.minimize = minimize
