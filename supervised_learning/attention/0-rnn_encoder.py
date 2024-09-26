#!/usr/bin/env python3
"""
RNNEncoder module for machine translation encoding using TensorFlow.
"""

import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """
    RNNEncoder class that inherits from TensorFlow's Keras Layer.
    Encodes input sequences for machine translation using GRU and embedding layers.
    """

    def __init__(self, vocab, embedding, units, batch):
        """
        Initializes the RNNEncoder.
        """
        super(RNNEncoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab, output_dim=embedding)
        self.gru = tf.keras.layers.GRU(
                units,
                return_sequences=True,
                return_state=True,
                recurrent_initializer='glorot_uniform'
                )

    def initialize_hidden_state(self):
        """
        Initializes the hidden states for the RNN cell to a tensor of zeros.
        """
        return tf.zeros((self.batch, self.units))

    def call(self, x, initial):
        """
        Forward pass through the encoder.
        """
        x = self.embedding(x)
        outputs, hidden = self.gru(x, initial_state=initial)
        return outputs, hidden
