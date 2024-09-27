#!/usr/bin/env python3
"""
Encoder module for the transformer.
"""

import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock


class Encoder(tf.keras.layers.Layer):
    """
    Encoder class that inherits from TensorFlow's Keras Layer.
    """

    def __init__(self, N, dm, h, hidden, input_vocab, max_seq_len, drop_rate=0.1):
        """
        Initializes the Encoder.
        """
        super(Encoder, self).__init__()
        self.N = N
        self.dm = dm

        # Embedding layer
        self.embedding = tf.keras.layers.Embedding(input_dim=input_vocab, output_dim=dm)

        # Positional encoding
        self.positional_encoding = positional_encoding(max_seq_len, dm)

        # Encoder blocks
        self.blocks = [EncoderBlock(dm, h, hidden, drop_rate) for _ in range(N)]

        # Dropout layer
        self.dropout = tf.keras.layers.Dropout(rate=drop_rate)

    def call(self, x, training, mask):
        """
        Forward pass for the Encoder.
        """
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)  # Apply embedding
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))  # Scale embeddings
        x += self.positional_encoding[:seq_len]

        x = self.dropout(x, training=training)  # Apply dropout to positional encoding

        for block in self.blocks:
            x = block(x, training, mask)  # Pass through each EncoderBlock

        return x
