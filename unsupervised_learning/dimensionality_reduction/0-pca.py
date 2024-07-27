#!/usr/bin/env python3
"""
PCA on a dataset
"""
import numpy as np


def pca(X, var=0.95):
    """
    Function that performs PCA on a dataset
    """
    # Calculate the SVD of input data
    U, S, V = np.linalg.svd(X, full_matrices=False)

    # Calculate the cumulative sum of the variance ratio
    var_ratio = np.cumsum(S**2) / np.sum(S**2)

    # Determine the number of components to keep
    nb_comp = np.argmax(var_ratio >= var) + 1

    # Select first nb_comp components
    W = V[:nb_comp].T

    return W
