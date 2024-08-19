#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
import GPyOpt
import matplotlib.pyplot as plt

# Load and preprocess data
data = load_breast_cancer()
X = data.data
y = data.target
X = StandardScaler().fit_transform(X)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model building function
def build_model(learning_rate, units, dropout_rate, l2_reg):
    model = Sequential()
    model.add(Dense(units=units, activation='relu', input_shape=(X_train.shape[1],),
                    kernel_regularizer=l2(l2_reg)))
    model.add(Dropout(rate=dropout_rate))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# Define the function to optimize
def model_score(params):
    learning_rate = float(params[:, 0])
    units = int(params[:, 1])
    dropout_rate = float(params[:, 2])
    l2_reg = float(params[:, 3])
    batch_size = int(params[:, 4])

    model = build_model(learning_rate, units, dropout_rate, l2_reg)
    
    checkpoint_path = f'checkpoint_lr_{learning_rate}_units_{units}_dropout_{dropout_rate}_l2_{l2_reg}_batch_{batch_size}.h5'
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
        ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, verbose=1)
    ]
    
    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        batch_size=batch_size,
                        epochs=50,
                        callbacks=callbacks,
                        verbose=0)
    
    val_acc = np.max(history.history['val_accuracy'])
    return -val_acc

# Define the bounds of the hyperparameters
bounds = [
    {'name': 'learning_rate', 'type': 'continuous', 'domain': (1e-5, 1e-1)},
    {'name': 'units', 'type': 'discrete', 'domain': (16, 32, 64, 128, 256)},
    {'name': 'dropout_rate', 'type': 'continuous', 'domain': (0.0, 0.5)},
    {'name': 'l2_reg', 'type': 'continuous', 'domain': (1e-6, 1e-2)},
    {'name': 'batch_size', 'type': 'discrete', 'domain': (16, 32, 64, 128)}
]

# Perform Bayesian Optimization
optimizer = GPyOpt.methods.BayesianOptimization(f=model_score, domain=bounds)
optimizer.run_optimization(max_iter=30)

# Save the optimization report
with open('bayes_opt.txt', 'w') as f:
    f.write(f'Best parameters found: {optimizer.x_opt}\n')
    f.write(f'Best validation accuracy: {-optimizer.fx_opt}\n')

# Plot convergence
optimizer.plot_convergence()
plt.savefig('convergence_plot.png')
plt.show()

