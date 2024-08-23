#!/usr/bin/env python3
"""
Variational Autoencoder
"""

import tensorflow.keras as keras
import tensorflow.keras.backend as K


def sampling(args):
    """
    Reparameterization trick by sampling from an isotropic unit Gaussian.
    Instead of sampling from Q(z|X), sample epsilon = N(0,I)
    and shift by the learned parameters mu and sigma.
    """
    mu, log_var = args
    batch = K.shape(mu)[0]
    dim = K.int_shape(mu)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return mu + K.exp(0.5 * log_var) * epsilon

def build_encoder(input_dims, hidden_layers, latent_dims):
    """
    Builds the encoder part of the autoencoder.
    """
    inputs = keras.Input(shape=(input_dims,))
    x = inputs
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)

    # Latent variables: mean and log variance
    mu = keras.layers.Dense(latent_dims, activation=None)(x)
    log_var = keras.layers.Dense(latent_dims, activation=None)(x)

    # Sampling layer using the reparameterization trick
    z = keras.layers.Lambda(sampling, output_shape=(latent_dims,))([mu, log_var])

    return keras.Model(inputs, [z, mu, log_var], name='encoder')

def build_decoder(latent_dims, hidden_layers, output_dims):
    """
    Builds the decoder part of the autoencoder.
    """
    latent_inputs = keras.Input(shape=(latent_dims,))
    x = latent_inputs
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)
    outputs = keras.layers.Dense(output_dims, activation='sigmoid')(x)
    return keras.Model(latent_inputs, outputs, name='decoder')

def vae_loss(inputs, outputs, mu, log_var, input_dims):
    """
    Custom loss function for Variational Autoencoder, combining reconstruction
    loss with KL divergence.
    """
    reconstruction_loss = keras.losses.binary_crossentropy(inputs, outputs)
    reconstruction_loss *= input_dims
    kl_loss = 1 + log_var - K.square(mu) - K.exp(log_var)
    kl_loss = -0.5 * K.sum(kl_loss, axis=-1)
    return K.mean(reconstruction_loss + kl_loss)

def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder.
    """
    encoder = build_encoder(input_dims, hidden_layers, latent_dims)
    decoder = build_decoder(latent_dims, hidden_layers, input_dims)

    inputs = keras.Input(shape=(input_dims,))
    z, mu, log_var = encoder(inputs)
    outputs = decoder(z)

    # Autoencoder model
    auto = keras.Model(inputs, outputs, name='vae')
    auto.add_loss(vae_loss(inputs, outputs, mu, log_var, input_dims))
    auto.compile(optimizer='adam')

    return encoder, decoder, auto
