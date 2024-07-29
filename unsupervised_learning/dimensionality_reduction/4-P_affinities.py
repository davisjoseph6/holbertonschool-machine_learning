#!/usr/bin/env python3
"""
P affinities (t-SNE)
"""
import numpy as np

P_init = __import__('2-P_init').P_init
HP = __import__('3-entropy').HP


def P_affinities(X, tol=1e-5, perplexity=30.0):
    """
    Calculates the symmetric P affinities of a dataset
    """
    D, P, betas, H = P_init(X, perplexity)
    n, d = X.shape

    # Binary search to find beta
    for i in range(n):
        beta_min = -np.inf
        beta_max = np.inf

        # Distance Di pairwise distances between a data point
        # and all other points except itself
        Di = D[i, np.concatenate((np.r_[:i], np.r_[i + 1:n]))]

        Hi, Pi = HP(Di, betas[i])
        # Evaluate whether perplexity is witin tolerance
        H_diff = Hi - H
        tries = 0

        while np.abs(H_diff) > tol and tries < 50:
            # If not, increase or decrease precision
            if H_diff > 0:
                beta_min = betas[i].copy()
                if beta_max == np.inf or beta_max == -np.inf:
                    betas[i] = betas[i] * 2
                else:
                    betas[i] = (betas[i] + beta_max) / 2
            else:
                beta_max = betas[i].copy()
                if beta_min == np.inf or beta_min == -np.inf:
                    betas[i] /= 2
                else:
                    betas[i] = (betas[i] + beta_min) / 2

            # Recompute entropy and affinities
            Hi, Pi = HP(Di, betas[i])
            H_diff = Hi - H
            tries += 1

        # Set final row of P, inserting the missing spot as 0
        P[i, np.concatenate((np.r_[:i], np.r_[i + 1:n]))] = Pi

    # Normalize
    P = (P.T + P) / (2 * n)

    return P
