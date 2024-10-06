#!/usr/bin/env python3
"""
Train a Transformer model for machine translation.
"""

import tensorflow as tf
from 3-dataset import Dataset
from 4-create_masks import create_masks

# Import the Transformer from 5-transformer.py
Transformer = __import__('5-transformer').Transformer

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, dm, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.dm = dm
        self.dm = tf.cast(self.dm, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.dm) * tf.math.minimum(arg1, arg2)

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))  # Ignore padding tokens
    loss_ = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)

def accuracy_function(real, pred):
    accuracies = tf.equal(real, tf.argmax(pred, axis=2))
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)
    accuracies = tf.cast(accuracies, dtype=tf.float32)
    return tf.reduce_sum(accuracies) / tf.reduce_sum(tf.cast(mask, dtype=tf.float32))

def train_transformer(N, dm, h, hidden, max_len, batch_size, epochs):
    """
    Trains a transformer model for machine translation of Portuguese to English.
    """
    # Load dataset
    dataset = Dataset(batch_size=batch_size, max_len=max_len)

    # Input and target vocab size
    input_vocab_size = dataset.tokenizer_pt.vocab_size + 2
    target_vocab_size = dataset.tokenizer_en.vocab_size + 2

    # Initialize Transformer model
    transformer = Transformer(N, dm, h, hidden, input_vocab_size, target_vocab_size, max_len)

    # Learning rate and Adam optimizer
    learning_rate = CustomSchedule(dm)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    # Metrics
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

    # Training step function
    @tf.function
    def train_step(inputs, target):
        target_input = target[:, :-1]
        target_real = target[:, 1:]
        encoder_mask, combined_mask, decoder_mask = create_masks(inputs, target_input)

        with tf.GradientTape() as tape:
            predictions = transformer(inputs, target_input, True, encoder_mask, combined_mask, decoder_mask)
            loss = loss_function(target_real, predictions)

        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

        train_loss(loss)
        train_accuracy(accuracy_function(target_real, predictions))

    # Training loop
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}')
        batch = 0

        for (inputs, target) in dataset.data_train:
            train_step(inputs, target)

            if batch % 50 == 0:
                print(f'Epoch {epoch + 1}, Batch {batch}: Loss {train_loss.result():.4f}, Accuracy {train_accuracy.result():.4f}')
            batch += 1

        print(f'Epoch {epoch + 1}: Loss {train_loss.result():.4f}, Accuracy {train_accuracy.result():.4f}')
        train_loss.reset_states()
        train_accuracy.reset_states()

    return transformer
