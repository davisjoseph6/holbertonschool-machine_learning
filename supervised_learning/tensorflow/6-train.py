#!/usr/bin/env python3
"""
Builds, trains, and saves a neural network classifier.
"""

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

# Import necessary custom functions
calculate_accuracy = _import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations, alpha,
          iterations, save_path="/tmp/model.ckpt"):
    """
    Builds, trains, and saves a neural network classifier.
    """
    # Create the placeholders
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])

    # Forward propagation
    y_pred = forward_prop(x, layer_sizes, activations)

    # Cost and accuracy
    loss = calculate_loss(y, y_pred)
    accuracy = calculate_accuracy(y, y_pred)

    # Training operation
    train_op = create_train_op(loss, alpha)

    # Add tensors and operations to collections to retrieve them later
    tf.add_to_collection('placeholders', x)
    tf.add_to_collection('placeholders', y)
    tf.add_to_collection('outputs', y_pred)
    tf.add_to_collection('losses', loss)
    tf.add_to_collection('accuracies', acuuracy)
    tf.add_to_collection('train_op', train_op)

    # Initialize all variables
    init = tf.global_variables_initializer()

    # Saver object to save the model
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)

        for i in range(iterations + 1):
            # Run training step
            sess.run(train_op, feed_dict={x: X_train, y: Y_train})

            if i % 100 == 0 or i == iterations:
                # Calculate loss and accuracy for both training and validation
                # sets
                train_cost, train_acc = sess.run([loss, accuracy], feed_dict={x: X_train, y: Y_train})
                valid_cost, valid_acc = sess.run([loss, accuracy], feed_dict={x: X_valid, y: Y_valid})

                # Print the cost and accuracy
                print(f"After {i} iterations:")
                print(f"\tTraining Cost: {train_cost}")
                print(f"\tTraining Accuracy: {train_acc}")
                print(f"\tValidation Cost: {valid_cost}")
                print(f"\tValidation Accuracy: {valid_acc}")

        # Save the model
        save_path = saver.save(sess, save_path)

    return save_path
