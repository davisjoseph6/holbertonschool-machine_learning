#!/usr/bin/env python3
"""
Calculates the F1 score for each class in a confusion matrix.
"""

import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
    Calculates the F1 score for each class in a confusion matrix.
    """
    # Calculate the sensitivity for each class
    sensitivities = sensitivity(confusion)

    # Calculate the precision for each class
    precisions = precision(confusion)

    # Calculate the F1 score for each class
    f1_scores = 2 * (precisions * sensitivities) / (precisions + sensitivities)

    return f1_scores
