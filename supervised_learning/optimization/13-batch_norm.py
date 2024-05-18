#!/usr/bin/env python3
"""
Normalize an unactivated output of a neural network using batch normalization
"""

import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    Normalizes an unactivated output of a neural network using batch
    normalization
    """
    # Calculate mean and variance
    mean = np.mean(Z, axis=0)
    variance = np.var(Z, axis=0)

    # Normalize Z
    Z_norm = (Z - mean) / np.sqrt(variance + epsilon)

    # Scale and shift
    Z_tilde = gamma * Z_norm + beta

    return Z_tilde
