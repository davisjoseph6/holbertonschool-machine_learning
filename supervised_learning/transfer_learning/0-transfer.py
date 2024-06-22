#!/usr/bin/env python3

from tensorflow import keras as K
import numpy as np

def preprocess_data(X, Y):
    X_p = K.applications.resnet50.preprocess_input(X.astype('float32'))
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p

def main():
    (X_train, Y_train), (X_test, Y_test) = K.datasets.cifar10.load_data()
    X_train, Y_train = preprocess_data(X_train, Y_train)
    X_test, Y_test = preprocess_data(X_test, Y_test)

    base_model = K.applications.ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3)
    )

    base_model.trainable = False

    model = K.models.Sequential([
        K.layers.Lambda(lambda image: K.backend.resize_images(image, 7, 7, 'channels_last'),
                        input_shape=(32, 32, 3)),
        base_model,
        K.layers.GlobalAveragePooling2D(),
        K.layers.Dense(256, activation='relu'),
        K.layers.Dropout(0.5),
        K.layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        X_train, Y_train,
        epochs=20,
        validation_data=(X_test, Y_test),
        batch_size=128
    )

    model.save('cifar10.h5')

if __name__ == "__main__":
    main()

