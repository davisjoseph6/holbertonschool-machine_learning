#!/usr/bin/env python3
"""
Creates a learning rate decay operation in TensorFlow using inverse time
decay.
"""

import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, decay_step):
    """
    Creates a learning rate decay operation in TensorFlow using inverse time
    decay.
    """
    return tf.keras.optimizers.schedules.InverseTimeDecay(
            initial_learning_rate=alpha,
            decay_steps=decay_step,
            decay_rate=decay_rate,
            staircase=True
            )
