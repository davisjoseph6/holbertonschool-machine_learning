#!/usr/bin/env python3
"""
Calculates the accuracy of a prediction.
"""

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()


def calculate_accuracy(y, y_pred):
    """
    Calculates the accuracy of a prediction
    """
    # Convert logits to label indexes (highest logit)
    correct_pred = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return accuracy
