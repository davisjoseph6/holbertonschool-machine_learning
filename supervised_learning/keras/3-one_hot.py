#!/usr/bin/env python3
"""
Converts a label vector into a one-hot matrix
"""

import numpy as np


def one_hot(labels, classes=None):
    """
    Converts a label vector into a one-hot matrix

    Args:
        labels (np.ndarray): Array of labels to be converted
        classes (int, optional): Number of classes. If not provided, inferred
        from labels.

    Returns:
        np.ndarray: One-hot matrix
    """
    if classes is None:
        classes = np.max(labels) + 1
    return np.eye(classes)[labels]
