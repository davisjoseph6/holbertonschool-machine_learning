#!/usr/bin/env python3
"""
t-SNE transformation
"""
import numpy as np
pca = __import__('1-pca').pca
P_affinities = __import__('4-P_affinities').P_affinities
grads = __import__('6-grads').grads
cost = __import__('7-cost').cost


def tsne(X, ndims=2, idims=50, perplexity=30.0, iterations=1000, lr=500):
    """
    Performs a t-SNE transformation
    """
    # Step 1: Reduce the dimensionality of X using PCA
    X_pca = pca(X, idims)

    # Step 2: Compute the pairwise affinities P using a perplexity of 30.0
    P = P_affinities(X_pca, perplexity=perplexity)

    # Step 3: Initialize Y randomly in the new dimensionality
    n, d = X.shape
    Y = np.random.randn(n, ndims)

    # Step 4: Early exaggeration
    P *= 4

    # Initialize the gain and previous update for momentum
    Y_momentum = np.zeros_like(Y)

    for i in range(iterations):
        # Compute gradients
        dY, Q = grads(Y, P)

        # Update the Y values
        if i < 20:
            momentum = 0.5
        else:
            momentum = 0.8

        Y_momentum = momentum * Y_momentum - lr * dY
        Y += Y_momentum

        # Re-center Y
        Y -= np.mean(Y, axis=0)

        # Every 100 iterations, print the cost
        if (i + 1) % 100 == 0:
            C = cost(P, Q)
            print(f"Cost at iteration {i + 1}: {C}")

        # Stop early exaggeration after the first 100 iterations
        if i == 99:
            P /= 4

    return Y
