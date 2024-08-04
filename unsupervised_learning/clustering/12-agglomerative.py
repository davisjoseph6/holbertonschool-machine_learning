#!/usr/bin/env python3
"""
Agglomerative clustering with Ward linkage
"""

import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """
    Performs agglomerative clustering on a dataset.
    """
    # Perform the hierarchical/agglomerative clustering using Ward's method
    Z = sch.linkage(X, method='ward')

    # Create the dendrogram
    sch.dendrogram(Z, color_threshold=dist)

    # Display the dendrogram
    plt.show()

    # Determine the cluster labels
    clss = sch.fcluster(Z, t=dist, criterion='distance')

    return clss
