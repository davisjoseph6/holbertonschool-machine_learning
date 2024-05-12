#!/usr/bin/env python3


import numpy as np
import tensorflow.compat.v1 as tf

def evaluate(X, Y, save_path):
    """
    Evaluates the output of a neural network.

    Parameters:
    X (numpy.ndarray): Input data to evaluate.
    Y (numpy.ndarray): One-hot labels for X.
    save_path (str): Location to load the model from.

    Returns:
    tuple: (predictions, accuracy, loss)
    """
    # Disable eager execution to use TensorFlow 1.x features
    tf.disable_eager_execution()

    # Create placeholders for inputs and labels
    x = tf.placeholder(tf.float32, [None, X.shape[1]])
    y = tf.placeholder(tf.float32, [None, Y.shape[1]])

    # Correctly define your model architecture to match the checkpoint
    layer = tf.layers.dense(inputs=x, units=256, activation=tf.nn.relu, name='layer')
    layer_1 = tf.layers.dense(inputs=layer, units=256, activation=tf.nn.relu, name='layer_1')
    logits = tf.layers.dense(inputs=layer_1, units=10, activation=None, name='layer_2')

    # Apply softmax to logits
    y_pred = tf.nn.softmax(logits)

    # Calculate loss
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits))

    # Calculate accuracy
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Create a saver object to load the model
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # Load the saved model
        saver.restore(sess, save_path)

        # Run the session to get predictions, accuracy, and loss
        predictions, acc, cost = sess.run([y_pred, accuracy, loss], feed_dict={x: X, y: Y})

    return predictions, acc, cost

