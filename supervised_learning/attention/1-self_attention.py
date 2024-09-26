#!/usr/bin/env python3
"""
SelfAttention module for machine translation using TensorFlow.
"""

import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """
    SelfAttention class that inherits from TensorFlow's Keras Layer.
    Computes the attention for machine translation based on the paper.
    """

    def __init__(self, units):
        """
        Initializes the SelfAttention.
        """
        super(SelfAttention, self).__init__()
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """
        Forward pass to calculate attention.
        """
        # Expand s_prev to have the same time steps as hidden_states
        s_prev_expanded = tf.expand_dims(s_prev, 1)

        # Calculate the score e_t using W, U, and V
        score = self.V(tf.nn.tanh(self.W(s_prev_expanded) + self.U(hidden_states)))

        # Calculate the attention weights using softmax
        weights = tf.nn.softmax(score, axis=1)

        # Calculate the context vector as the weighted sum of hidden_states
        context = tf.reduce_sum(weights * hidden_states, axis=1)

        return context, weights
