#!/usr/bin/env python3
"""
PCA on a dataset
"""
import numpy as np


def pca(X, ndim):
    """
    Function that performs PCA on a dataset
    """
    # Center the dataset by subtracting the mean
    X_centered = X - np.mean(X, axis=0)

    # Calculate the SVD of the centered data
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    # Select the first ndim components
    W = Vt[:ndim].T

    # Compute the transformed data
    T = np.dot(X_centered, W)

    return T
