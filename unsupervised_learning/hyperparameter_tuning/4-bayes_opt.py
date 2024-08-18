#!/usr/bin/env python3
"""
Module documentation for '4-bayes_opt'.
"""

import numpy as np
from scipy.stats import norm
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

    def acquisition(self):
        """
        Calculate the next best sample location using the Expected Improvement
        (EI) acquisition function.
        """
        # Predict the mean and standard deviation for the sample points
        mu, sigma = self.gp.predict(self.X_s)

        # Ensure sigma is a 1D array
        sigma = sigma.flatten()

        if self.minimize:
            mu_sample_opt = np.min(self.gp.Y)
            improvement = mu_sample_opt - mu - self.xsi
        else:
            mu_sample_opt = np.max(self.gp.Y)
            improvement = mu - mu_sample_opt - self.xsi

        # Calculate the expected improvement
        with np.errstate(divide='warn'):
            Z = improvement / sigma
            EI = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
            EI[sigma == 0.0] = 0.0

        # The next best sample is the one with the maximum expected improvement
        X_next = self.X_s[np.argmax(EI)].reshape(1,)

        return X_next, EI
