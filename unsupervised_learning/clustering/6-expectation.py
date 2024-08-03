#!/usr/bin/env python3
"""
Calculates the expectation step in the EM algorithm for a GMM.
"""

import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    Calculates the expectation step in the EM algorithm for a GMM.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None
    if not isinstance(pi, np.ndarray) or pi.ndim != 1:
        return None, None
    if not isinstance(m, np.ndarray) or m.ndim != 2:
        return None, None
    if not isinstance(S, np.ndarray) or S.ndim != 3:
        return None, None
    if (X.shape[1] != m.shape[1] or S.shape[1] != S.shape[2] or
            m.shape[1] != S.shape[1] or pi.shape[0] != m.shape[0]):
        return None, None

    n, d = X.shape
    k = pi.shape[0]

    # Calculate the PDF for each cluster
    likelihoods = np.zeros((k, n))
    for i in range(k):
        likelihoods[i] = pdf(X, m[i], S[i])

    # Calculate the posterior probabilities (responsibilities)
    weighted_likelihoods = pi[:, np.newaxis] * likelihoods
    total_likelihood = np.sum(weighted_likelihoods, axis=0)
    g = weighted_likelihoods / total_likelihood

    # Calculate the log likelihood
    log_likelihood = np.sum(np.log(total_likelihood))

    return g, log_likelihood
