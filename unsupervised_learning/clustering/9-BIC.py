#!/usr/bin/env python3
"""
Finds the best number of clusters for a GMM using the BIC
"""

import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Finds the best number of clusters for a GMM using the Bayesian Information Criterion (BIC).
    """
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or
            not isinstance(kmin, int) or kmin <= 0 or
            (kmax is not None and (not isinstance(kmax, int) or kmax < kmin)) or
            not isinstance(iterations, int) or iterations <= 0 or
            not isinstance(tol, float) or tol < 0 or
            not isinstance(verbose, bool)):
        return None, None, None, None

    n, d = X.shape

    if kmax is None:
        kmax = n

    log_likelihoods = []
    bics = []

    for k in range(kmin, kmax + 1):
        pi, m, S, g, log_likelihood = expectation_maximization(X, k, iterations, tol, verbose)
        if pi is None or m is None or S is None or g is None or log_likelihood is None:
            return None, None, None, None

        p = k * d + k * d * (d + 1) // 2 + k - 1  # Number of parameters
        bic = p * np.log(n) - 2 * log_likelihood

        log_likelihoods.append(log_likelihood)
        bics.append(bic)

    log_likelihoods =  np.array(log_likelihoods)
    bics = np.array(bics)

    best_k = kmin + np.argmin(bics)
    best_result = expectation_maximization(X, best_k, iterations, tol, verbose)

    return best_k, best_result[:3], log_likelihoods, bics
