#!/usr/bin/env python3
"""
L2 Regularization Cost.
"""

import tensorflow as tf


def l2_reg_cost(cost, model):
    """
    Calculats the cost of a neural network with L2 regularization.
    """
    return cost + tf.add_n(model.losses)
