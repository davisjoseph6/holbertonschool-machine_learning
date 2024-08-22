#!/usr/bin/env python3
"""
This module contains the implementation of a sparse autoencoder model using
TensorFlow.
"""

import tensorflow.keras as keras
from tensorflow.keras import regularizers


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """
    Create a sparse autoencoder with the specified dimensions and
    L1 regularization.
    """
    # Encoder
    inputs = keras.Input(shape=(input_dims,))
    encoded = inputs
    for nodes in hidden_layers:
        encoded = keras.layers.Dense(nodes, activation='relu')(encoded)
    latent = keras.layers.Dense(latent_dims, activation='relu',
                                activity_regularizer=regularizers.L1(lambtha))(encoded)

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

