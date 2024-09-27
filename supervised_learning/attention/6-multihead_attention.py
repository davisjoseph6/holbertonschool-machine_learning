#!/usr/bin/env python3
"""
MultiHeadAttention module for performing multi-head attention using TensorFlow.
"""

import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    MultiHeadAttention class that performs multi-head attention.
    """

    def __init__(self, dm, h):
        """
        Initializes the MultiHeadAttention.
        """
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.dm = dm
        self.depth = dm // h

        # Initialize the Dense layers for Q, K, V
        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)

        # Linear layer to generate the final attention output
        self.linear = tf.keras.layers.Dense(dm)

    def split_heads(self, x, batch_size):
        """
        Splits the last dimension into (h, depth).
        """
        x = tf.reshape(x, (batch_size, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask):
        """
        Forward pass through the MultiHeadAttention layer.
        """
        batch_size = tf.shape(Q)[0]

        # Pass inputs through dense layers
        Q = self.Wq(Q)  # Shape (batch_size, seq_len_q, dm)
        K = self.Wk(K)  # Shape (batch_size, seq_len_v, dm)
        V = self.Wv(V)  # Shape (batch_size, seq_len_v, dm)

        # Split Q, K, V into multiple heads

        # Shape (batch_size, h, seq_len_q, depth)
        Q = self.split_heads(Q, batch_size)

        # Shape (batch_size, h, seq_len_v, depth)
        K = self.split_heads(K, batch_size)

        # Shape (batch_size, h, seq_len_v, depth)
        V = self.split_heads(V, batch_size)

        # Apply scaled dot-product attention
        scaled_attention, weights = sdp_attention(Q, K, V, mask)

        # Concatenate attention output
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1,
                                                         self.dm))

        # Final linear layer
        output = self.linear(concat_attention)

        return output, weights
