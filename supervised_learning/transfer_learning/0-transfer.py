#!/usr/bin/env python3

from tensorflow import keras as K
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16
import numpy as np

def preprocess_data(X, Y):
    """Preprocesses the CIFAR 10 data for your model."""
    X_p = X.astype('float32') / 255.0
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p

def create_model():
    """Creates and compiles the model for CIFAR 10 classification."""
    # Load the pre-trained VGG16 model
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False

    # Create new model on top of the output of the base model
    model = K.Sequential()
    model.add(layers.Lambda(lambda image: K.backend.resize_images(image, 7, 7, "channels_last"), input_shape=(32, 32, 3)))
    model.add(base_model)
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def main():
    """Main function to train the model and save it."""
    # Load and preprocess the CIFAR 10 data
    (X_train, Y_train), (X_test, Y_test) = K.datasets.cifar10.load_data()
    X_train, Y_train = preprocess_data(X_train, Y_train)
    X_test, Y_test = preprocess_data(X_test, Y_test)

    # Create and train the model
    model = create_model()
    model.fit(X_train, Y_train, epochs=10, batch_size=128, validation_data=(X_test, Y_test))

    # Save the model
    model.save('cifar10.h5')

if __name__ == '__main__':
    main()

