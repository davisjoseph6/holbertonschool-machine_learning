#!/usr/bin/env python3
"""
Determines the steady state probabilities of a regular Markov chain.
"""

import numpy as np


def regular(P):
    """
    Determines the steady state probabilities of a regular Markov chain.
    """
    if (not isinstance(P, np.ndarray) or P.ndim != 2 or
            P.shape[0] != P.shape[1]):
        return None

    n = P.shape[0]

    # Check if the matrix is regular by ensuring some power of P has all
    # positive entries
    # We check P^k for some small k
    P_power = np.linalg.matrix_power(P, n)
    if not np.all(P_power > 0):
        return None

    # To find the steady state, solve the equation πP = π
    # Which can be rewritten as (P.T - I)π.T = 0
    # With the additional constraint that the sum of π equals 1
    A = P.T - np.eye(n)
    A = np.vstack((A, np.ones((1, n))))
    b = np.zeros((n + 1,))
    b[-1] = 1

    try:
        steady_state = np.linalg.lstsq(A, b, rcond=None)[0]
        steady_state = steady_state.reshape((1, n))
        return steady_state
    except np.linalg.LinAlgError:
        return None
