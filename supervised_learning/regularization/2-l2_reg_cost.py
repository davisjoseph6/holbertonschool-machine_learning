#!/usr/bin/env python3
"""
Calculates the cost of a neural network with L2 regularization.
"""

import tensorflow as tf


def l2_reg_cost(cost, model):
    """
    Calculats the cost of a neural network with L2 regularization.
    """
    regularization_loss = tf.add_n(model.losses)
    total_cost = cost + regularization_loss
    return total_cost
