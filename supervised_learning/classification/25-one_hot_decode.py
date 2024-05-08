#!/usr/bin/env python3
"""
a function that converts a numeric label vector into a one-hot matrix
"""

import numpy as np


def one_hot_decode(one_hot):
    """
    Converts a one-hot matrix into a vector of labels.
    """
    if not isinstance(one_hot, np.ndarray) or one_hot.ndim != 2:
        return None
    try:
        labels = np.argmax(one_hot, axis=0)
        return labels
    except Exception as e:
        return None
