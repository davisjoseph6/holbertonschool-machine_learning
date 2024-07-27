#!/usr/bin/env python3
"""
t-SNE Shannon entropy and P affinities calculation
"""
import numpy as np


def HP(Di, beta):
    """
    Calculates the Shannon entropy and P affinities relative to a data point
    """
    # Compute the P affinities
    Pi = np.exp(-Di * beta)
    sum_Pi = np.sum(Pi)
    Pi = Pi / sum_Pi

    # Compute the Shannon entropy
    Hi = -np.sum(Pi * np.log2(Pi))

    return Hi, Pi
