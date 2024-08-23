#!/usr/bin/env python3
"""
This module contains the implementation of a convolutional autoencoder using TensorFlow.
"""

import tensorflow.keras as keras

def autoencoder(input_dims, filters, latent_dims):
    """
    Creates a convolutional autoencoder with the specified dimensions.

    Parameters
    ----------
    input_dims : tuple
        A tuple of integers containing the dimensions of the model input.
    filters : list
        A list containing the number of filters for each convolutional layer in the encoder.
        The filters should be reversed for the decoder.
    latent_dims : tuple
        A tuple of integers containing the dimensions of the latent space representation.

    Returns
    -------
    encoder : keras.Model
        The encoder model.
    decoder : keras.Model
        The decoder model.
    auto : keras.Model
        The full autoencoder model.
    """
    # Encoder
    inputs = keras.Input(shape=input_dims)
    x = inputs
    for f in filters:
        x = keras.layers.Conv2D(f, (3, 3), activation='relu', padding='same')(x)
        x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)

    # Latent space
    latent = x

    # Decoder
    decoded_input = keras.Input(shape=latent_dims)
    x = decoded_input
    for f in reversed(filters[:-1]):
        x = keras.layers.Conv2D(f, (3, 3), activation='relu', padding='same')(x)
        x = keras.layers.UpSampling2D((2, 2))(x)

    # Conv2D layers for adjusting filters and size
    x = keras.layers.Conv2D(filters[-1], (3, 3), activation='relu', padding='same')(x)
    x = keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = keras.layers.UpSampling2D((2, 2))(x)

    # Final layer to match desired output
    outputs = keras.layers.Conv2D(input_dims[-1], (3, 3), activation='sigmoid', padding='same')(x)

    # Models
    encoder = keras.Model(inputs, latent, name='encoder')
    decoder = keras.Model(decoded_input, outputs, name='decoder')
    auto = keras.Model(inputs, decoder(encoder(inputs)), name='autoencoder')

    # Compile the autoencoder model
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto

