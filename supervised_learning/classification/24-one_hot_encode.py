#!/usr/bin/env python3
"""
a function that converts a numeric label vector into a one-hot matrix
"""

import numpy as np


def one_hot_encode(Y, classes):
    """
    Converts a numeric label vector Y into a one-hot matrix
    """
    if not isinstance(Y, np.ndarray) or len(Y) == 0:
        return None
    if not isinstance(classes, int) or classes <= 0:
        return None

    try:
        one_hot = np.eye(classes)[Y].T
        return one_hot
    except IndexError as e:
        return None
