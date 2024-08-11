#!/usr/bin/env python3
"""
Determines the probability of a Markov chain.
"""

import numpy as np


def markov_chain(P, s, t=1):
    """
    Determines the probability of a Markov chain being in a particular state.
    """
    if not isinstance(P, np.ndarray) or not isinstance(s, np.ndarray):
        return None
    if P.ndim != 2 or s.ndim != 2:
        return None
    if P.shape[0] != P.shape[1] or s.shape[0] != 1 or s.shape[1] != P.shape[0]:
        return None
    if not isinstance(t, int) or t < 1:
        return None

    # Calculate the probability after t iterations
    state_prob = np.matmul(s, np.linalg.matrix_power(P, t))

    return state_prob
