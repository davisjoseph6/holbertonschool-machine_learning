#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import GPyOpt
import os

# Define the function to optimize
def create_model(learning_rate, num_units, dropout_rate, l2_reg, batch_size):
    model = Sequential([
        Dense(num_units, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg), input_shape=(input_shape,)),
        Dropout(dropout_rate),
        Dense(1, activation='linear')
        ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mean_squared_error')
    return model

def train_model(params):
    learning_rate, num_units, dropout_rate, l2_reg, batch_size = params[0]
    model = create_model(learning_rate, int(num_units), dropout_rate, l2_reg, int(batch_size))

    # Checkpoint filename based on hyperparameters
    checkpoint_name = f'model_lr{learning_rate}_units{int(num_units)}_dropout{dropout_rate}_l2{l2_reg}_batch{int(batch_size)}.h5'
    checkpoint = ModelCheckpoint(checkpoint_name, save_best_only=True, monitor='val_loss', mode='min', verbose=1)
    early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)

    history = model.fit(X_train, y_train,
                        epochs=100,
                        batch_size=int(batch_size),
                        validation_split=0.2,
                        callbacks=[checkpoint, early_stop],
                        verbose=0)

    val_loss = np.min(history.history['val_loss'])
    return val_loss

# Define the hyperparameter space
space = [
        {'name': 'learning_rate', 'type': 'continuous', 'domain': (1e-5, 1e-2)},
        {'name': 'num_units', 'type': 'discrete', 'domain': (32, 64, 128, 256)},
        {'name': 'dropout_rate', 'type': 'continuous', 'domain': (0.0, 0.5)},
        {'name': 'l2_reg', 'type': 'continuous', 'domain': (1e-6, 1e-2)},
        {'name': 'batch_size', 'type': 'discrete', 'domain': (16, 32, 64, 128)}
        ]

# Load your dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.boston_housing.load_data()
input_shape = X_train.shape[1]

# Create the Bayesian optimization object
optimizer = GPyOpt.methods.BayesianOptimization(f=train_model, domain=space, acquisition_type='EI', maximize=False)

# Run optimization
optimizer.run_optimization(max_iter=30)

# Save optimization report
with open('bayes_opt.txt', 'w') as f:
    f.write(f'Optimized parameters: {optimizer.x_opt}\n')
    f.write(f'Best validation loss: {optimizer.fx_opt}\n')

# Plot convergence
optimizer.plot_convergence()
