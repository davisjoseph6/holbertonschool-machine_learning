#!/usr/bin/env python3
"""
Module documentation for '2-gp'.
"""

import numpy as np


class GaussianProcess:
    """
    Class that represents a noiseless 1D Gaussian process.
    """

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        Class constructor.
        """

        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """
        Calculates the covariance kernel matrix between two matrices using
        the Radial Basis Function (RBF) kernel.
        """
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + \
            np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
        return self.sigma_f**2 * np.exp(-0.5 / self.l**2 * sqdist)

    def predict(self, X_s):
        """
        Predicts the mean and standard deviation of points in a
        Gaussian process.
        """
        # Calculate the covariance between X_s and X
        K_s = self.kernel(X_s, self.X)
        # Calculate the covariance matrix of the points in X_s
        K_ss = self.kernel(X_s, X_s)
        # Calculate the inverse of the covariance matrix K
        K_inv = np.linalg.inv(self.K)

        # Compute the mean vector mu for the points in X_s
        mu_s = K_s.dot(K_inv).dot(self.Y).flatten()

        # Compute the covariance matrix Sigma for the points in X_s
        sigma_s = K_ss - K_s.dot(K_inv).dot(K_s.T)

        # The variance is the diagonal of the covariance matrix
        sigma_s_diag = np.diag(sigma_s)

        return mu_s, sigma_s_diag

    def update(self, X_new, Y_new):
        """
        Updates the Gaussian Process with a new sample point.
        """
        # Update X and Y with the new data point
        self.X = np.vstack((self.X, X_new.reshape(-1, 1)))
        self.Y = np.vstack((self.Y, Y_new.reshape(-1, 1)))

        # Calculate the covariance between the new point and all previous
        # points
        K_s = self.kernel(self.X, X_new.reshape(-1, 1))

        # Update the K matrix with the new row and column
        K_new_row = K_s[:-1, :]
        K_new_entry = K_s[-1, :].reshape(1, -1)

        self.K = np.block(
                [
            [self.K, K_new_row],
            [K_new_row.T, K_new_entry]
            ]
                )
