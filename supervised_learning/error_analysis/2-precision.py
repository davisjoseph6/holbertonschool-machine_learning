#!/usr/bin/env python3
"""
Calculates the precision for each class in a confusion matrix
"""

import numpy as np


def precision(confusion):
    """
    Calculates the precision for each class in a confusion matrix.
    """
    # True Positives are the diagonal elements of the confusion matrix
    true_positives = np.diag(confusion)

    # False Postives are the sum of each column, excluding the diagonal element
    false_positives = np.sum(confusion, axis=0) - true_positives

    # Precision is calculated as True Positives /
    # (True Positives + False Positives)
    precision = true_positives / (true_positives + false_positives)

    return precision
