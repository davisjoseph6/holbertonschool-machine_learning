#!/usr/bin/env python3
"""
Defines the Transformer model for machine translation.
"""

import sys
sys.path.append('../attention')

Encoder = __import__('9-transformer_encoder').Encoder
Decoder = __import__('10-transformer_decoder').Decoder

import tensorflow as tf
from tensorflow.keras.layers import Dense

class Transformer(tf.keras.Model):
    """
    Transformer model for sequence-to-sequence tasks with N encoder and decoder blocks.
    """
    def __init__(self, N, dm, h, hidden, input_vocab_size, target_vocab_size, max_len, dropout_rate=0.1):
        """
        Initializes the Transformer model.
        """
        super(Transformer, self).__init__()
        self.encoder = Encoder(N, dm, h, hidden, input_vocab_size, max_len, dropout_rate)
        self.decoder = Decoder(N, dm, h, hidden, target_vocab_size, max_len, dropout_rate)
        self.final_layer = Dense(target_vocab_size)

    def call(self, inputs, target, training, encoder_mask, look_ahead_mask, decoder_mask):
        """
        Forward pass for the Transformer model.
        """
        enc_output = self.encoder(inputs, training, encoder_mask)  # Encoder output
        dec_output = self.decoder(target, enc_output, training, look_ahead_mask, decoder_mask)  # Decoder output
        final_output = self.final_layer(dec_output)  # Final linear layer

        return final_output
