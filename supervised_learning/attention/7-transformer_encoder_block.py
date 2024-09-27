#!/usr/bin/env python3
"""
EncoderBlock module for creating an encoder block for a transformer using
TensorFlow.
"""

import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    """
    EncoderBlock class that creates an encoder block for a transformer.
    """

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        Initializes the EncoderBlock.
        """
        super(EncoderBlock, self).__init__()

        # Multi-head attention layer
        self.mha = MultiHeadAttention(dm, h)

        # Fully connected layers
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)

        # Layer normalization layers
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # Dropout layers
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """
        Forward pass through the encoder block.
        """
        # Multi-head attention layer
        attn_output, _ = self.mha(x, x, x, mask)

        # Dropout and add & normalize for the first sublayer
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        # Feed forward network
        dense_output = self.dense_hidden(out1)
        ffn_output = self.dense_output(dense_output)

        # Dropout and add & normalize for the second sublayer
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2
