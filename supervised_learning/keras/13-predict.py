#!/usr/bin/env python3
"""
Makes a predictions using a neural network
"""

import tensorflow.keras as K


def predict(network, data, verbose=False):
    """
    Makes a prediction using a neural network.
    """
    return network.predict(data, verbose=verbose)
