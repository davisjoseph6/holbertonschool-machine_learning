#!/usr/bin/env python3
"""
Module to load a FrozenLake environment with customization options.
"""

import gymnasium as gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """
    Loads the pre-made FrozenLakeEnv environment from gymnasium.
    """
    env = gym.make("FrozenLake-v1",
                   desc=desc,
                   map_name=map_name,
                   is_slippery=is_slippery)
    return env
