#!/usr/bin/env python3
"""
RNNDecoder module for machine translation decoding using TensorFlow.
"""

import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """
    RNNDecoder class that inherits from TensorFlow's Keras Layer.
    Decodes for machine translation using GRU and attention mechanism.
    """

    def __init__(self, vocab, embedding, units, batch):
        """
        Initializes the RNNDecoder.
        """
        super(RNNDecoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab,
                                                   output_dim=embedding)
        self.gru = tf.keras.layers.GRU(units=units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

        self.F = tf.keras.layers.Dense(vocab)
        self.attention = SelfAttention(units)

    def call(self, x, s_prev, hidden_states):
        """
        Forward pass through the decoder.
        """
        # Calculate the context vector using self-attention
        context, _ = self.attention(s_prev, hidden_states)

        # Embed the input x
        x = self.embedding(x)

        # Concatenate the context vector with the embedded input
        x = tf.concat([tf.expand_dims(context, 1), x], axis=-1)

        # Pass the concatenated input through the GRU layer
        output, s = self.gru(x)

        # Remove the extra axis
        output = tf.squeeze(output, axis=1)

        # Pass the GRU output through the Dense layer to predict the next word
        y = self.F(output)

        return y, s
