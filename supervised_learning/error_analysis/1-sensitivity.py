#!/usr/bin/env python3
"""
Calculates the sensitivity for each class in a confusion matrix.
"""

import numpy as np


def sensitivity(confusion):
    """
    Calculates the sensitivity for each class in a confusion matrix.
    """
    # True Positives are the diagonal elements of the confusion matrix
    true_positives = np.diag(confusion)

    # False Negatives are the sum of each row, excluding the diagonal element
    false_negatives = np.sum(confusion, axis=1) - true_positives

    # Sensitivity is calculated as True Positives /
    # (True Positives + False Negatives)
    sensitivity = true_positives / (true_positives + false_negatives)

    return sensitivity
