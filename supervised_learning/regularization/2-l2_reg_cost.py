#!/usr/bin/env python3
"""
L2 Regularization Cost.
"""

import tensorflow as tf

def l2_reg_cost(cost, model):
    """
    Calculates the cost of a neural network with L2 regularization.

    Parameters:
    cost (tensor): Tensor containing the cost of the network without L2 regularization.
    model (tf.keras.Model): Keras model that includes layers with L2 regularization.

    Returns:
    tensor: Tensor containing the total cost for each layer of the network, accounting for L2 regularization.
    """
    # Collect the regularization losses for each layer
    regularization_losses = model.losses
    
    # Add the regularization losses to the cost
    total_cost = cost + tf.add_n(regularization_losses)
    
    # Return a tensor containing the total cost for each layer
    return tf.convert_to_tensor([cost.numpy()] + [loss.numpy() for loss in regularization_losses])

