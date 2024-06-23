#!/usr/bin/env python3

from tensorflow import keras as K
import numpy as np

def preprocess_data(X, Y):
    X_p = K.applications.convnext.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p

def resize_and_compute_features(model, X, batch_size=50, target_size=(224, 224), save_path="features.npy"):
    num_images = X.shape[0]
    features = []
    for start in range(0, num_images, batch_size):
        end = min(start + batch_size, num_images)
        X_batch = X[start:end]
        X_resized = np.array([K.preprocessing.image.smart_resize(img, target_size) for img in X_batch])
        batch_features = model.predict(X_resized)
        features.append(batch_features)
        print(f'Processed batch {start // batch_size + 1}/{(num_images + batch_size - 1) // batch_size}')
    features = np.vstack(features)
    np.save(save_path, features)
    return features

if __name__ == "__main__":
    print("Loading CIFAR-10 data...")
    (X_train, Y_train), (X_test, Y_test) = K.datasets.cifar10.load_data()
    X_train_p, Y_train_p = preprocess_data(X_train, Y_train)
    X_test_p, Y_test_p = preprocess_data(X_test, Y_test)

    print("Loading pre-trained ConvNeXtSmall model...")  # Using a smaller model
    base_model = K.applications.ConvNeXtSmall(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3)
    )

    print("Freezing base model layers...")
    for layer in base_model.layers:
        layer.trainable = False

    try:
        print("Loading precomputed training features...")
        train_features = np.load('train_features.npy')
    except FileNotFoundError:
        print("Computing features for the training set...")
        train_features = resize_and_compute_features(base_model, X_train_p, batch_size=25, save_path='train_features.npy')

    try:
        print("Loading precomputed validation features...")
        val_features = np.load('val_features.npy')
    except FileNotFoundError:
        print("Computing features for the validation set...")
        val_features = resize_and_compute_features(base_model, X_test_p, batch_size=25, save_path='val_features.npy')

    print("Building new model...")
    model = K.Sequential([
        K.layers.InputLayer(input_shape=train_features.shape[1:]),
        K.layers.Flatten(),
        K.layers.Dense(256, activation='relu'),
        K.layers.Dropout(0.5),
        K.layers.Dense(10, activation='softmax')
    ])

    print("Compiling the model...")
    model.compile(
        optimizer=K.optimizers.Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("Training the model...")
    model.fit(
        train_features, Y_train_p,
        epochs=10,
        validation_data=(val_features, Y_test_p),
        batch_size=16  # Reduced batch size for training
    )

    print("Saving the model to cifar10.h5...")
    model.save('cifar10.h5')
    print("Model saved successfully.")

