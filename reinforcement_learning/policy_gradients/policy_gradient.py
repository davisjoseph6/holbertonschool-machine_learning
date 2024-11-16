#!/usr/bin/env python3
"""
This module provides the policy_gradient function, which computes the
action and its gradient for reinforcement learning using policy gradients.
"""

import numpy as np


def policy(matrix, weight):
    """
    Computes the policy using a softmax function over the dot product of
    the state and weight matrix.
    """
    z = np.dot(matrix, weight)  # Linear combination of input and weights
    exp = np.exp(z - np.max(z))  # Softmax with stability adjustment
    return exp / exp.sum(axis=1, keepdims=True)


def policy_gradient(state, weight):
    """
    Computes the Monte-Carlo policy gradient based on a state
    and weight matrix.
    """
    # Reshape the state to fit the dimensions expected by the policy function
    state = state.reshape(1, -1)

    # Compute policy (action probabilities)
    probs = policy(state, weight).flatten()

    # Select action by sampling from the probability distribution
    action = np.random.choice(len(probs), p=probs)

    # Compute the gradient of the log probability of the action
    dsoftmax = probs.copy()
    dsoftmax[action] -= 1  # Gradient of the log-softmax for chosen action
    gradient = np.outer(state, -dsoftmax)

    return action, gradient
