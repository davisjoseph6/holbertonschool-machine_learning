#!/usr/bin/env python3
"""
This module contains the policy function that computes a policy matrix using weights.
"""

import numpy as np

def policy(matrix, weight):
    """
    Computes the policy using a softmax function over the dot product of the state and weight matrix.
    
    Args:
        matrix (np.ndarray): The state matrix.
        weight (np.ndarray): The weight matrix.
    
    Returns:
        np.ndarray: The computed policy as a probability distribution over actions.
    """
    z = np.dot(matrix, weight)  # Linear combination of input and weights
    exp = np.exp(z - np.max(z))  # Softmax with stability adjustment
    return exp / exp.sum(axis=1, keepdims=True)

