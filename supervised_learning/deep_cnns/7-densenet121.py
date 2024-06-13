#!/usr/bin/env python3
"""
    DenseNet-121
"""

import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
        builds the DenseNet-121 architecture as described in
        'Densely Connected Convolutional Networks'

        :param growth_rate: growth rate
        :param compression: compression factor
        input shape (224, 224, 3)
        conv preceded by batchNorm and rectified ReLU
        He normal initialization

        :return: keras model
    """
    # define initialization
    initializer = K.initializers.HeNormal()
    input_data = K.Input(shape=(224, 224, 3))

    # First conv
    batchN1 = K.layers.BatchNormalization()(input_data)
    relu1 = K.layers.Activation(activation='relu')(batchN1)
    nbr_filter = growth_rate * 2
    conv1 = K.layers.Conv2D(filters=nbr_filter,
                            kernel_size=(7, 7),
                            strides=(2, 2),
                            padding='same',
                            kernel_initializer=initializer)(relu1)
    pool1 = K.layers.MaxPooling2D(pool_size=(3, 3),
                                  strides=(2, 2),
                                  padding='same')(conv1)

    # Dense blocks 1
    db1, nbr_filter = dense_block(pool1, nbr_filter, growth_rate, 6)
    trans_l1, nbr_filter = transition_layer(db1, nbr_filter, compression)

    # Dense blocks 2
    db2, nbr_filter = dense_block(trans_l1, nbr_filter, growth_rate, 12)
    trans_l2, nbr_filter = transition_layer(db2, nbr_filter, compression)

    # Dense blocks 3
    db3, nbr_filter = dense_block(trans_l2, nbr_filter, growth_rate, 24)
    trans_l3, nbr_filter = transition_layer(db3, nbr_filter, compression)

    # Dense blocks 4 (without transition layer)
    db4, nbr_filter = dense_block(trans_l3, nbr_filter, growth_rate, 16)

    # fully connected layer
    pool_g = K.layers.AveragePooling2D(pool_size=(7, 7))(db4)
    full_connected = K.layers.Dense(units=1000,
                                    activation="softmax",
                                    kernel_initializer=initializer)(pool_g)

    model = K.models.Model(input_data, full_connected)

    return model
