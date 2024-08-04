#!/usr/bin/env python3
"""
Expectation-Maximization for GMM
"""

import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
    Performs the expectation maximization for a GMM.
    """
    if (
            not isinstance(X, np.ndarray) or X.ndim != 2
            or not isinstance(k, int) or k <= 0
            or not isinstance(iterations, int) or iterations <= 0
            or not isinstance(tol, float) or tol < 0
            or not isinstance(verbose, bool)
            ):

        return None, None, None, None, None

    # Initialize priors, centroid means, and covariance matrices
    pi, m, S = initialize(X, k)

    for i in range(iterations):
        # Evaluate the probabilities and likelihoods with current parameters
        g, prev_li = expectation(X, pi, m, S)

        # In verbose mode, print the likelihood every 10 iterations after 0
        if verbose and i % 10 == 0:
            print(f"Log Likelihood after {i} iterations: {round(prev_li, 5)}")

        # Re-estimate the parameters with the new values
        pi, m, S = maximization(X, g)

        # Evaluate new log likelihood
        g, li = expectation(X, pi, m, S)

        # If the likelihood varied by less than the tolerance value, we stop
        if np.abs(li - prev_li) <= tol:
            break

    # Last verbose message with current likelihood
    if verbose:
        # NOTE i + 1 since it has been updated once more since last print
        print(f"Log Likelihood after {i + 1} iterations: {round(li, 5)}")

    return pi, m, S, g, li
