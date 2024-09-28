#!/usr/bin/env python3
"""
Decoder module for the transformer.
"""

import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
DecoderBlock = __import__('8-transformer_decoder_block').DecoderBlock


class Decoder(tf.keras.layers.Layer):
    """
    Decoder class that inherits from TensorFlow's Keras Layer.
    """

    def __init__(self, N, dm, h, hidden, target_vocab, max_seq_len,
                 drop_rate=0.1):
        """
        Initializes the Decoder.
        """
        super(Decoder, self).__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(input_dim=target_vocab,
                                                   output_dim=dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [
                DecoderBlock(dm, h, hidden, drop_rate) for _ in range(N)
                ]
        self.dropout = tf.keras.layers.Dropout(rate=drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """
        Forward pass for the Decoder.
        """
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)  # Apply embedding
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))  # Scale embeddings
        x += self.positional_encoding[:seq_len]

        # Apply dropout to positional encoding
        x = self.dropout(x, training=training)

        for block in self.blocks:
            x = block(x,
                      encoder_output,
                      training, look_ahead_mask,
                      padding_mask)

        return x
