#!/usr/bin/env python3
"""
Transformer module implementation.
"""

import tensorflow as tf
Encoder = __import__('9-transformer_encoder').Encoder
Decoder = __import__('10-transformer_decoder').Decoder


class Transformer(tf.keras.Model):
    """
    Transformer class inheriting from TensorFlow's Keras Model.
    """

    def __init__(self, N, dm, h, hidden, input_vocab, target_vocab,
                 max_seq_input, max_seq_target, drop_rate=0.1):
        """
        Initializes the Transformer model.
        """
        super(Transformer, self).__init__()
        self.encoder = Encoder(N, dm, h, hidden, input_vocab,
                               max_seq_input, drop_rate)
        self.decoder = Decoder(N, dm, h, hidden, target_vocab,
                               max_seq_target, drop_rate)
        self.linear = tf.keras.layers.Dense(units=target_vocab)

    def call(self, inputs, target, training, encoder_mask,
             look_ahead_mask, decoder_mask):
        """
        Forward pass for the Transformer model.
        """
        # Encoder output
        encoder_output = self.encoder(inputs,
                                      training,
                                      encoder_mask)

        # Decoder output
        decoder_output = self.decoder(target,
                                      encoder_output,
                                      training,
                                      look_ahead_mask,
                                      decoder_mask)

        output = self.linear(decoder_output)  # Final linear transformation

        return output
