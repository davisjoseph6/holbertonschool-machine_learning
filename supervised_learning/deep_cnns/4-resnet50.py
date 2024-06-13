#!/usr/bin/env python3
"""
ResNet-50 Architecture
"""

from tensorflow import keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
    Builds the ResNet-50 architecture as described in
    'Deep Residual Learning for Image Recognition' (2015).

    Returns:
    - The Keras model
    """
    init = K.initializers.HeNormal(seed=0)

    input_layer = K.Input(shape=(224, 224, 3))

    # Initial convolution and max pooling layers
    conv1 = K.layers.Conv2D(64,
                            (7, 7),
                            strides=(2, 2),
                            padding='same',
                            kernel_initializer=init)(input_layer)
    norm1 = K.layers.BatchNormalization(axis=3)(conv1)
    act1 = K.layers.Activation('relu')(norm1)
    max_pool1 = K.layers.MaxPooling2D((3, 3),
                                      strides=(2, 2),
                                      padding='same')(act1)

    # First set of blocks
    proj_block1 = projection_block(max_pool1, [64, 64, 256], s=1)
    id_block1_1 = identity_block(proj_block1, [64, 64, 256])
    id_block1_2 = identity_block(id_block1_1, [64, 64, 256])

    # Second set of blocks
    proj_block2 = projection_block(id_block1_2, [128, 128, 512], s=2)
    id_block2_1 = identity_block(proj_block2, [128, 128, 512])
    id_block2_2 = identity_block(id_block2_1, [128, 128, 512])
    id_block2_3 = identity_block(id_block2_2, [128, 128, 512])

    # Third set of blocks
    proj_block3 = projection_block(id_block2_3, [256, 256, 1024], s=2)
    id_block3_1 = identity_block(proj_block3, [256, 256, 1024])
    id_block3_2 = identity_block(id_block3_1, [256, 256, 1024])
    id_block3_3 = identity_block(id_block3_2, [256, 256, 1024])
    id_block3_4 = identity_block(id_block3_3, [256, 256, 1024])
    id_block3_5 = identity_block(id_block3_4, [256, 256, 1024])

    # Fourth set of blocks
    proj_block4 = projection_block(id_block3_5, [512, 512, 2048], s=2)
    id_block4_1 = identity_block(proj_block4, [512, 512, 2048])
    id_block4_2 = identity_block(id_block4_1, [512, 512, 2048])

    # Average pooling and output layer
    avg_pool = K.layers.AveragePooling2D((7, 7),
                                         strides=(1, 1))(id_block4_2)
    output_layer = K.layers.Dense(1000,
                                  activation='softmax',
                                  kernel_initializer=init)(avg_pool)

    # Create the model
    model = K.models.Model(inputs=input_layer,
                           outputs=output_layer)

    return model
