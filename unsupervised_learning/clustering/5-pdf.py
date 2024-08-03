#!/usr/bin/env python3
"""
Calculates the probability density function of a Gaussian distribution.
"""

import numpy as np


def pdf(X, m, S):
    """
    Calculates the probability density function of a Gaussian distribution.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    if not isinstance(m, np.ndarray) or m.ndim != 1:
        return None
    if not isinstance(S, np.ndarray) or S.ndim != 2:
        return None
    if (X.shape[1] != m.shape[0] or S.shape[0] != S.shape[1] or
            S.shape[0] != m.shape[0]:
        return None

    n, d = X.shape

    # Calculate the determinant and inverse of the covariance matrix
    try:
        det_S = np.linalg.det(S)
        inv_S = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        return None

    if det_S <= 0:
        return None

    norm_factor = 1 / np.sqrt((2 * np.pi) ** d * det_S)
    diff = X - m
    exp_term = np.exp(-0.5 * np.sum(diff @ inv_S * diff, axis=1))

    P = norm_factor * exp_term
    P = np.maximum(P, 1e-300)

    return P
