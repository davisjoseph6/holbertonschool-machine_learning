#!/usr/bin/env python3
"""
    PCA on a dataset
"""
import numpy as np


def pca(X, var=0.95):
    """
    Function that performs PCA on a dataset

    :param X: numpy.ndarray of shape (n, d) where:
        - n is the number of data points
        - d is the number of dimensions in each point
        - all dimensions have a mean of 0 across all data points
    :param var: the  fraction of the variance that the PCA transformation
        should maintain

    :return: the weights matrix, W,
        that maintains var fraction of X‘s original variance
    """

    # Calculate the SVD of input data
    U, S, V = np.linalg.svd(X, full_matrices=False)

    # Calculate the cumulative sum of the variance ratio
    var_ratio = np.cumsum(S**2) / np.sum(S**2)

    # Determine the number of components to keep
    nb_comp = np.argmax(var_ratio >= var) + 1

    # select first nb_comp
    W = V[:nb_comp + 1].T

    return W
