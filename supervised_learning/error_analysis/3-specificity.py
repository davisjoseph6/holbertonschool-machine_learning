#!/usr/bin/env python3
"""
Calculates the specificity for each class in a confusion matrix.
"""

import numpy as np


def specificity(confusion):
    """
    Calculates the specificity for each class in a confusion matrix.
    """
    # Total number of classes
    classes = confusion.shape[0]

    # True Positives are the diagonal elements of the confusion matrix
    true_positives = np.diag(confusion)

    # False Positives are the sum of each column, excluding the diagonal element
    false_positives = np.sum(confusion, axis=0) - true_positives

    # False Negatives are the sum of each row, excluding the diagonal element
    false_negatives = np.sum(confusion, axis=1) - true_positives

    # True Negatives are calculated by subtracting the sums of false positives,
    # false negatives, and true positives from the total sum
    true_negatives = np.sum(confusion) - (true_positives + false_positives + false_negatives)

    # Specificity is calculated as True Negatives / (True Negatives + False
    # False Positives)
    specificity = true_negatives / (true_negatives + false_positives)

    return specificity
