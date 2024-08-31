#!/usr/bin/env python3

from tensorflow.keras import layers, models

def convolutional_GenDiscr():
    def get_generator():
        # Generator model
        inputs = layers.Input(shape=(16,))
        x = layers.Dense(2048)(inputs)
        x = layers.Reshape((2, 2, 512))(x)
        x = layers.UpSampling2D()(x)
        x = layers.Conv2D(64, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('tanh')(x)
        x = layers.UpSampling2D()(x)
        x = layers.Conv2D(16, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('tanh')(x)
        x = layers.UpSampling2D()(x)
        x = layers.Conv2D(1, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        outputs = layers.Activation('tanh')(x)
        return models.Model(inputs, outputs, name="generator")

    def get_discriminator():
        # Discriminator model
        inputs = layers.Input(shape=(16, 16, 1))
        x = layers.Conv2D(32, (3, 3), padding='same')(inputs)
        x = layers.MaxPooling2D()(x)
        x = layers.Activation('tanh')(x)
        x = layers.Conv2D(64, (3, 3), padding='same')(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Activation('tanh')(x)
        x = layers.Conv2D(128, (3, 3), padding='same')(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Activation('tanh')(x)
        x = layers.Conv2D(256, (3, 3), padding='same')(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Activation('tanh')(x)
        x = layers.Flatten()(x)
        outputs = layers.Dense(1)(x)
        return models.Model(inputs, outputs, name="discriminator")

    return get_generator(), get_discriminator()

