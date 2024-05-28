#!/usr/bin/env python3
"""
Trains a model using mini-batch gradient descent with
optional validation data, early stopping, and learning rate decay
"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent
    """
    callbacks = []

    # Early stopping
    if early_stopping and validation_data is not None:
        early_stopping_callback = K.callbacks.EarlyStopping(
                monitor='val_loss', patience=patience,
                restore_best_weights=True
                )
        callbacks.append(early_stopping_callback)

        # Learning rate decay
    if learning_rate_decay and validation_data is not None:
        def schedule(epoch):
            return alpha / (1 + decay_rate * epoch)

        lr_decay_callback = K.callbacks.LearningRateScheduler(
                schedule, verbose=1
                )
        callbacks.append(lr_decay_callback)

    history = network.fit(data, labels,
                          batch_size=batch_size,
                          epochs=epochs,
                          verbose=verbose,
                          shuffle=shuffle,
                          validation_data=validation_data,
                          callbacks=callbacks)
    return history
