#!/usr/bin/env python3
"""
Module to initialize the Q-table for a FrozenLake environment.
"""

import numpy as np


def q_init(env):
    """
    Initializes the Q-table as a numpy array of zeros.
    """
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    Q_table = np.zeros((num_states, num_actions))
    return Q_table
