#!/usr/bin/env python3
"""
Performs the backward algorithm for a hidden Markov model.
"""

import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """
    Performs the backward algorithm for a hidden Markov model.
    """

    try:
        T = Observation.shape[0]
        N = Emission.shape[0]

        # Initialize the backward path probabilities matrix
        B = np.zeros((N, T))

        # Initialization step
        B[:, T-1] = 1

        # Recursion step
        for t in range(T-2, -1, -1):
            for i in range(N):
                B[i, t] = np.sum(Transition[i, :] *
                                 Emission[:, Observation[t+1]] *
                                 B[:, t+1])

        # Termination step
        P = np.sum(Initial.T * Emission[:, Observation[0]] * B[:, 0])

        return P, B

    except Exception:
        return None, None
