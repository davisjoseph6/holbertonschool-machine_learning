#!/usr/bin/env python3
"""
Builds, trains, and saves a neural network classifier.
"""

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

# Import the necessary modules
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop

def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations,
          alpha, iterations, save_path="/tmp/model.ckpt"):
    """
    Builds, trains, and saves a neural network classifier.

    Arguments:
    X_train: numpy.ndarray containing the training input data
    Y_train: numpy.ndarray containing the training labels
    X_valid: numpy.ndarray containing the validation input data
    Y_valid: numpy.ndarray containing the validation labels
    layer_sizes: list containing the number of nodes in each layer
    activations: list containing the activation functions for each layer
    alpha: the learning rate
    iterations: the number of iterations to train over
    save_path: designates where to save the model

    Returns:
    the path where the model was saved
    """
    nx = X_train.shape[1]
    classes = Y_train.shape[1]

    # Create placeholders for inputs and labels
    x, y = create_placeholders(nx, classes)

    # Build the forward propagation network
    y_pred = forward_prop(x, layer_sizes, activations)

    # Compute the loss and accuracy
    loss = calculate_loss(y, y_pred)
    accuracy = calculate_accuracy(y, y_pred)

    # Create the training operation
    train_op = create_train_op(loss, alpha)

    # Initialize all the variables
    init_op = tf.global_variables_initializer()

    # Create a saver object to save the model
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init_op)

        # Training loop
        for i in range(iterations + 1):
            # Run the training step and compute loss and accuracy for training data
            _, train_loss, train_accuracy = sess.run([train_op, loss, accuracy], feed_dict={x: X_train, y: Y_train})

            # Every 100 iterations, evaluate the model on training and validation data
            if i % 100 == 0 or i == iterations:
                valid_loss, valid_accuracy = sess.run([loss, accuracy], feed_dict={x: X_valid, y: Y_valid})

                print(f"After {i} iterations:")
                print(f"\tTraining Cost: {train_loss}")
                print(f"\tTraining Accuracy: {train_accuracy}")
                print(f"\tValidation Cost: {valid_loss}")
                print(f"\tValidation Accuracy: {valid_accuracy}")

        # Save the model at the end of training
        save_path = saver.save(sess, save_path)

    return save_path
