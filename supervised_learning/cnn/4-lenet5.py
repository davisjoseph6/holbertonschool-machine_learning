#!/usr/bin/env python3
"""
Builds a modified version of the LeNet-5 architecture using tensorflow.
"""

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def lenet5(x, y):
    """
    Builds a modified version of the LeNet-5 architecture using tensorflow.
    """

    # Initialize the he_normal initializer
    initializer = tf.keras.initializers.VarianceScaling(scale=2.0)

    # First Convolutional Layer
    conv1 = tf.layers.Conv2D(filters=6, kernel_size=(5, 5), padding='same',
                             activation=tf.nn.relu,
                             kernel_initializer=initializer)(x)

    # First Max Pooling Layer
    pool1 = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1)

    # Second Convolutional Layer
    conv2 = tf.layers.Conv2D(filters=16, kernel_size=(5, 5), padding='valid',
                             activation=tf.nn.relu,
                             kernel_initializer=initializer)(pool1)

    # Second Max Pooling Layer
    pool2 = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)

    # Flatten the output of the last pooling layer
    flatten = tf.layers.Flatten()(pool2)

    # Fully Connected Layer with 120 nodes
    fc1 = tf.layers.Dense(units=120, activation=tf.nn.relu,
                          kernel_initializer=initializer)(flatten)

    # Fully Connected Layer with 84 nodes
    fc2 = tf.layers.Dense(units=84, activation=tf.nn.relu,
                          kernel_initializer=initializer)(fc1)

    # Fully Connected Softmax Output Layer with 10 nodes
    logits = tf.layers.Dense(units=10, kernel_initializer=initializer)(fc2)
    y_pred = tf.nn.softmax(logits)

    # Loss function
    loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=logits)

    # Training operation
    train_op = tf.train.AdamOptimizer().minimize(loss)

    # Accuracy calculation
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return y_pred, train_op, loss, accuracy
