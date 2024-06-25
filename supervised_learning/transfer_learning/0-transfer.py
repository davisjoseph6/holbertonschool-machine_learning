#!/usr/bin/env python3

from tensorflow import keras as K
import numpy as np


def preprocess_data(X, Y):
    X_p = K.applications.densenet.preprocess_input(X.astype('float32'))
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p

def main():
    # Load and preprocess CIFAR-10 dataset
    (X_train, Y_train), (X_test, Y_test) = K.datasets.cifar10.load_data()
    X_train, Y_train = preprocess_data(X_train, Y_train)
    X_test, Y_test = preprocess_data(X_test, Y_test)

    # Define the model
    base_model = K.applications.DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False

    inputs = K.Input(shape=(32, 32, 3))
    x = K.layers.Lambda(lambda image: K.backend.resize_images(image, 7, 7, "channels_last"))(inputs)
    x = base_model(x, training=False)
    x = K.layers.GlobalAveragePooling2D()(x)
    x = K.layers.Dense(256, activation='relu')(x)
    outputs = K.layers.Dense(10, activation='softmax')(x)

    model = K.Model(inputs, outputs)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, Y_train, epochs=10, batch_size=128, validation_data=(X_test, Y_test), verbose=1)

    # Save the model
    model.save('cifar10.h5')

if __name__ == '__main__':
    main()
