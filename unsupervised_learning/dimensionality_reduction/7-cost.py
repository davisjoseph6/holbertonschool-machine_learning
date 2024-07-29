#!/usr/bin/env python3
"""
t-SNE cost calculation
"""
import numpy as np


def cost(P, Q):
    """
    Calculates the cost of the t-SNE transformation
    """
    # Ensure no division by zero errors by taking the maximum of P, Q and
    # a small value
    P = np.maximum(P, 1e-12)
    Q = np.maximum(Q, 1e-12)

    # Calculate the cost using the Kullback-Leibler divergence
    C = np.sum(P * np.log(P / Q))

    return C
