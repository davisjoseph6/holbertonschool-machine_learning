#!/usr/bin/env python3

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

def evaluate(X, Y, save_path):
    """
    Evaluates the output of a neural network.
    """
    graph = tf.Graph()
    with graph.as_default():
        saver = tf.train.import_meta_graph(save_path + '.meta')
        
        # Checking and retrieving tensors
        try:
            x = tf.get_collection('x')[0]
            y = tf.get_collection('y')[0]
            y_pred = tf.get_collection('y_pred')[0]
            loss = tf.get_collection('loss')[0]
            accuracy = tf.get_collection('accuracy')[0]
        except IndexError:
            print("Failed to retrieve tensors. Check the collection names.")
            return None, None, None

    with tf.Session(graph=graph) as sess:
        saver.restore(sess, save_path)
        Y_pred_oh, loss_val, accuracy_val = sess.run([y_pred, loss, accuracy], feed_dict={x: X, y: Y})

    return Y_pred_oh, accuracy_val, loss_val

