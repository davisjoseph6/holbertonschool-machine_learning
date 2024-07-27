#!/usr/bin/env python3
"""
Performs PCA on a dataset.
"""

import numpy as np


def pca(X, var=0.95):
    """
    Performs PCA on a dataset.
    """
    # Compute the covariance matrix
    cov_matrix = np.cov(X, rowvar=False)

    # Compute the eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort the eigenvalues and corresponding eigenvectors in descending order
    sorted_idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_idx]
    eigenvectors = eigenvectors[:, sorted_idx]

    # Calculate the cumulative variance
    cumulative_variance = np.cumsum(eigenvalues) / np.sum(eigenvalues)

    # Find the number of components to retain the given variance
    num_components = np.searchsorted(cumulative_variance, var) + 1

    # Get the weight matrix
    W = eigenvectors[:, :num_components]

    return W
