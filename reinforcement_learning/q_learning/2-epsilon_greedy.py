#!/usr/bin/env python3
"""
Module to determine the next action using epsilon-greedy policy.
"""

import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Uses epsilon-greedy policy to select the next action.
    """
    # Decide whether to explore or exploit
    if np.random.uniform(0, 1) < epsilon:
        # Exploration: choose a random action
        action = np.random.randint(Q.shape[1])
    else:
        # Exploitation: choose the best action from Q-table
        action = np.argmax(Q[state])

    return action
