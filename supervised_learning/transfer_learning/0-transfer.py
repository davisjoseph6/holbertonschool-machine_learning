#!/usr/bin/env python3

from tensorflow import keras as K
import numpy as np
from tqdm import tqdm  # for progress bar

def preprocess_data(X, Y):
    """
    Pre-processes the data for your model

    Args:
    X: numpy.ndarray of shape (m, 32, 32, 3) containing the CIFAR 10 data, where m is the number of data points
    Y: numpy.ndarray of shape (m,) containing the CIFAR 10 labels for X

    Returns:
    X_p: numpy.ndarray containing the preprocessed X
    Y_p: numpy.ndarray containing the preprocessed Y
    """
    X_p = K.applications.convnext.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p

def resize_and_predict(model, X, batch_size=500, target_size=(224, 224)):
    """
    Resize images in batches and predict using the model to avoid OOM issues

    Args:
    model: the pre-trained model to use for prediction
    X: numpy.ndarray of shape (m, 32, 32, 3) containing the CIFAR 10 data
    batch_size: number of images to process at a time
    target_size: tuple of (height, width) to resize images to

    Returns:
    features: numpy.ndarray containing the model predictions
    """
    num_images = X.shape[0]
    features = []
    for start in tqdm(range(0, num_images, batch_size), desc="Processing batches"):
        end = min(start + batch_size, num_images)
        X_batch = X[start:end]
        X_resized = np.array([K.preprocessing.image.smart_resize(img, target_size) for img in X_batch])
        batch_features = model.predict(X_resized)
        features.append(batch_features)
    return np.vstack(features)

if __name__ == "__main__":
    # Load CIFAR-10 data
    (X_train, Y_train), (X_test, Y_test) = K.datasets.cifar10.load_data()
    X_train_p, Y_train_p = preprocess_data(X_train, Y_train)
    X_test_p, Y_test_p = preprocess_data(X_test, Y_test)

    # Load the pre-trained ConvNeXtXLarge model
    base_model = K.applications.ConvNeXtXLarge(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3)
    )

    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False

    # Precompute the output of the frozen layers for the training set
    train_features = resize_and_predict(base_model, X_train_p)

    # Precompute the output of the frozen layers for the validation set
    val_features = resize_and_predict(base_model, X_test_p)

    # Add new trainable layers on top of the frozen layers
    model = K.Sequential([
        K.layers.InputLayer(input_shape=train_features.shape[1:]),
        K.layers.Flatten(),
        K.layers.Dense(256, activation='relu'),
        K.layers.Dropout(0.5),
        K.layers.Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(
        optimizer=K.optimizers.Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train the model using the precomputed features
    model.fit(
        train_features, Y_train_p,
        epochs=20,
        validation_data=(val_features, Y_test_p),
        batch_size=128
    )

    # Save the model
    model.save('cifar10.h5')

