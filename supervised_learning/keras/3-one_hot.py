#!/usr/bin/env python3
"""
Converts a label vector into a one-hot matrix
"""

import tensorflow.keras as K


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
    one_hot_matrix = K.utils.to_categorical(labels, num_classes=classes)
    return one_hot_matrix
