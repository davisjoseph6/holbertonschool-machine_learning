#!/usr/bin/env python3
"""
K-means clustering
"""

import numpy as np
from sklearn.cluster import KMeans


def kmeans(X, k):
    """
    Performs K-means on a dataset
    """
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    C = kmeans.cluster_centers_
    clss = kmeans.labels_
    return C, clss
