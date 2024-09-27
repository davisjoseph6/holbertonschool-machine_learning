#!/usr/bin/env python3
"""
Encoder module for creating the encoder of a transformer using TensorFlow.
"""

import tensorflow as tf
import numpy as np
positional_encoding = __import__('4-positional_encoding').positional_encoding
EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock


class Encoder(tf.keras.layers.Layer):
    """
    Encoder class that creates the encoder for a transformer.
    """

    def __init__(self, N, dm, h, hidden, input_vocab, max_seq_len, drop_rate=0.1):
        """
        Initializes the Encoder.
        """
        super(Encoder, self).__init__()

        self.dm = dm
        self.N = N

        # Embedding layer
        self.embedding = tf.keras.layers.Embedding(input_dim=input_vocab, output_dim=dm)

        # Positional encoding
        self.positional_encoding = positional_encoding(max_seq_len, dm)

        # Encoder blocks
        self.blocks = [EncoderBlock(dm, h, hidden, drop_rate) for _ in range(N)]

        # Dropout layer
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """
        Forward pass for the Encoder.
        """
        input_seq_len = x.shape[1]

        # Apply embedding and scale by sqrt(dm)
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))

        # Add positional encoding
        x += self.positional_encoding[:input_seq_len]

        # Apply dropout
        x = self.dropout(x, training=training)

        # Pass through each encoder block
        for block in self.blocks:
            x = block(x, training, mask)

        return x
