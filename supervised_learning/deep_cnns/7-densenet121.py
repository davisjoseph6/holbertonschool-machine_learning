#!/usr/bin/env python3
"""
DenseNet-121 Implementation
"""

from tensorflow import keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    Builds the DenseNet-121 architecture as described in
    'Densely Connected Convolutional Networks'.

    Parameters:
    growth_rate (int): The growth rate for the dense blocks.
    compression (float): The compression factor for the transition layers.

    Returns:
    keras.Model: The Keras model representing DenseNet-121.
    """
    # Define the input layer with the specified shape
    input_0 = K.Input(shape=(224, 224, 3))

    # Initial batch normalization and ReLU activation
    BN_0 = K.layers.BatchNormalization()(input_0)
    ReLU_0 = K.layers.Activation(activation='relu')(BN_0)

    # Initial convolution with 64 filters, 7x7 kernel size, and stride of 2
    conv_0 = K.layers.Conv2D(
            filters=64,
            kernel_size=(7, 7),
            strides=(2, 2),
            kernel_initializer=K.initializers.he_normal(seed=0),
            padding="same")(ReLU_0)

    # Max pooling with 3x3 pool size and stride of 2
    pool_0 = K.layers.MaxPooling2D(pool_size=(3, 3),
                                   strides=2,
                                   padding="same")(conv_0)

    # First dense block with 6 layers
    out_1, nb = dense_block(pool_0, pool_0.shape[3], growth_rate, 6)
    # First transition layer
    trans_1, nb = transition_layer(out_1, nb, compression)

    # Second dense block with 12 layers
    out_2, nb = dense_block(trans_1, trans_1.shape[3], growth_rate, 12)
    # Second transition layer
    trans_2, nb = transition_layer(out_2, nb, compression)

    # Third dense block with 24 layers
    out_3, nb = dense_block(trans_2, trans_2.shape[3], growth_rate, 24)
    # Third transition layer
    trans_3, nb = transition_layer(out_3, nb, compression)

    # Fourth dense block with 16 layers.
    out_4, nb = dense_block(trans_3, trans_3.shape[3], growth_rate, 16)

    # Global average pooling layer
    avg_pooling = K.layers.AveragePooling2D(pool_size=(7, 7),
                                            padding="same")(out_4)

    # Fully connected layer with 1000 units and softmax
    # activation for classification
    dense = K.layers.Dense(units=1000,
                           activation='softmax')(avg_pooling)

    # Create the Keras model
    model = K.Model(input_0, dense)
    return model
