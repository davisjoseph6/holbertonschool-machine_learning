#!/usr/bin/env python3
"""
Creates a confusion matrix.
"""

import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Creates a confusion matrix
    """
    # Get the number of classes
    classes = labels.shape[1]

    # Initialize the confusion matrix with zeros
    confusion_matrix = np.zeros((classes, classes), dtype=int)

    # Convert one-hot encoded labels to class indices
    true_classes = np.argmax(labels, axis=1)
    predicted_classes = np.argmax(logits, axis=1)

    # Populate the confusion matrix
    for true, pred in zip(true_classes, predicted_classes):
        confusion_matrix[true, pred] += 1

    return confusion_matrix
