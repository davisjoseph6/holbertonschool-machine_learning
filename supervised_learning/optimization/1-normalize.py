#!/usr/bin/env python3
"""
Normalize (standardize) a matrix X.
"""

import numpy as np


def normalize(X, m, s):
    """
    Normalize (standardize) a matrix X.
    """
    return (X - m) / s
