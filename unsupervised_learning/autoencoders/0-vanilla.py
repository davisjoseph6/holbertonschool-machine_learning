#!/usr/bin/env python3
"""
This module contains the implementation of an autoencoder model using
TensorFlow.
"""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Create an autoencoder with the specified dimensions.
    """
    # Encoder
    inputs = keras.Input(shape=(input_dims,))
    encoded = inputs
    for nodes in hidden_layers:
        encoded = keras.layers.Dense(nodes, activation='relu')(encoded)
    latent = keras.layers.Dense(latent_dims, activation='relu')(encoded)

    # Decoder
    decoded_input = keras.Input(shape=(latent_dims,))
    decoded = decoded_input
    for nodes in reversed(hidden_layers):
        decoded = keras.layers.Dense(nodes, activation='relu')(decoded)
    outputs = keras.layers.Dense(input_dims, activation='sigmoid')(decoded)

    # Models
    encoder = keras.Model(inputs, latent, name='encoder')
    decoder = keras.Model(decoded_input, outputs, name='decoder')
    auto = keras.Model(inputs, decoder(encoder(inputs)), name='autoencoder')

    # Compile the autoencoder model
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
