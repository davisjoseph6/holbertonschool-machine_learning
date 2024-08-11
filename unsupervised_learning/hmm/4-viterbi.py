#!/usr/bin/env python3
"""
Calculates te most likely sequence of hidden states for a hidden markov model.
"""

import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """
    Calculates the most likely sequence of hidden states for a
    hidden markov model.
    """

    try:
        T = Observation.shape[0]
        N = Emission.shape[0]

        # Initialize the path probability matrix (delta) and the backpointer
        # matrix (psi)
        delta = np.zeros((N, T))
        psi = np.zeros((N, T), dtype=int)

        # Initialize the first column of delta and psi
        delta[:, 0] = Initial.T * Emission[:, Observation[0]]
        psi[:, 0] = 0

        # Recursion step
        for t in range(1, T):
            for j in range(N):
                prob = (delta[:, t-1] *
                        Transition[:, j] *
                        Emission[j, Observation[t]])
                delta[j, t] = np.max(prob)
                psi[j, t] = np.argmax(prob)

        # Termination step
        P = np.max(delta[:, T-1])
        path = np.zeros(T, dtype=int)
        path[T-1] = np.argmax(delta[:, T-1])

        # Path backtracking
        for t in range(T-2, -1, -1):
            path[t] = psi[path[t+1], t+1]

        return path.tolist(), P

    except Exception:
        return None, None
