#!/usr/bin/env python3
"""
Updates a variable using the RMSProp optimization algorithm.
"""

import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    Updates a variable using the RMSProp
    """
    s = beta2 * s + (1 - beta2) * np.square(grad)
    var = var - alpha * grad / (np.sqrt(s) + epsilon)
    return var, s
