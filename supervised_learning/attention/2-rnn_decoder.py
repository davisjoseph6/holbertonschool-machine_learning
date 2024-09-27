#!/usr/bin/env python3
"""
RNN Decoder
"""

import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """
    This class represents the decoder for machine translation.
    """

    def __init__(self, vocab, embedding, units, batch):
        """
        Initializes the RNNDecoder.

        Args:
            vocab (int): Size of the output vocabulary.
            embedding (int): Dimensionality of the embedding vector.
            units (int): Number of hidden units in the RNN cell.
            batch (int): Batch size.
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
        Forward pass for the RNN decoder with attention mechanism.

        Args:
            x (tf.Tensor): Tensor of shape `(batch, 1)` containing the previous
                word in the target sequence as an index of the target vocabulary.
            s_prev (tf.Tensor): Tensor of shape `(batch, units)` containing the
                previous decoder hidden state.
            hidden_states (tf.Tensor): Tensor of shape `(batch, input_seq_len, units)`
                containing the outputs of the encoder.

        Returns:
            y (tf.Tensor): Tensor of shape `(batch, vocab)` containing the output word
                as a one-hot vector in the target vocabulary.
            s (tf.Tensor): Tensor of shape `(batch, units)` containing the new decoder hidden state.
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

