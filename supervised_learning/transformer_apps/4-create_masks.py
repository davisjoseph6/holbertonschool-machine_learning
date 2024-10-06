#!/usr/bin/env python3
"""
Creates padding and lookahead masks for Transformer training/validation.
"""

import tensorflow as tf


def create_padding_mask(seq):
    """
    Creates a padding mask for a given sequence.
    """
    # Create a mask where padding tokens are represented as 1,
    # and valid tokens as 0
    mask = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # Add extra dimensions for broadcasting
    return mask[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(size):
    """
    Creates a lookahead mask to mask future tokens.
    """
    # Create a lower triangular matrix to mask future tokens
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # Upper triangular part will be 1, lower will be 0


def create_masks(inputs, target):
    """
    Creates all necessary masks for training/validation.
    """
    # Encoder padding mask
    encoder_mask = create_padding_mask(inputs)

    # Decoder padding mask to be applied in the second attention block
    decoder_mask = create_padding_mask(inputs)

    # Create a lookahead mask for the decoder's first attention block
    look_ahead_mask = create_look_ahead_mask(tf.shape(target)[1])

    # Create a padding mask for the target sequence
    target_padding_mask = create_padding_mask(target)

    # Combine the lookahead mask and target padding mask for the
    # decoder's first attention block
    combined_mask = tf.maximum(look_ahead_mask, target_padding_mask)

    return encoder_mask, combined_mask, decoder_mask
