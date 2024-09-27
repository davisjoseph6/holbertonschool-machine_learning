#!/usr/bin/env python3
"""
DecoderBlock module for creating a decoder block for a transformer using
TensorFlow.
"""

import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    """
    DecoderBlock class that creates a decoder block for a transformer.
    """

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        Initializes the DecoderBlock.
        """
        super(DecoderBlock, self).__init__()

        # First and second multi-head attention layers
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)

        # Fully connected layers
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)

        # Layer normalization layers
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # Dropout layers
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """
        Forward pass through the decoder block.
        """
        # First multi-head attention layer
        attn1, _ = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)

        # Residual connection + normalization
        out1 = self.layernorm1(x + attn1)

        # Second multi-head attention layer
        attn2, _ = self.mha2(out1,
                             encoder_output,
                             encoder_output,
                             padding_mask)
        attn2 = self.dropout2(attn2, training=training)

        # Residual connection + normalization
        out2 = self.layernorm2(out1 + attn2)

        # Feed forward network
        ffn_output = self.dense_hidden(out2)
        ffn_output = self.dense_output(ffn_output)
        ffn_output = self.dropout3(ffn_output, training=training)

        # Final layer normalization with residual connection
        output = self.layernorm3(out2 + ffn_output)

        return output
