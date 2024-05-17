#!/usr/bin/env python3
"""
Evaluates the output of a neural network
"""

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()


def evaluate(X, Y, save_path):
    """
    Evaluates the output of a neural network
    Args:
        X (np.ndarray): The input data to evaluate
        Y (np.ndarray): The one-hot labels for X
        save_path (str): The location to load the model from
    Returns:
        np.ndarray: The network's prediction
        float: The accuracy of the model
        float: The loss of the model
    """
    with tf.Session() as sess:
        # Load the saved model
        saver = tf.train.import_meta_graph(save_path + '.meta')
        saver.restore(sess, save_path)

        # Retrieve the necessary tensors from the graph
        graph = tf.get_default_graph()
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        y_pred = tf.get_collection('y_pred')[0]
        loss = tf.get_collection('loss')[0]
        accuracy = tf.get_collection('accuracy')[0]

        # Evaluate the model on the given data
        prediction, accuracy_value, loss_value = sess.run(
            [y_pred, accuracy, loss], feed_dict={x: X, y: Y})

    return prediction, accuracy_value, loss_value
