#!/usr/bin/env python3

from tensorflow import keras as K
import numpy as np

def preprocess_data(X, Y):
    """Preprocess the CIFAR-10 data."""
    X_p = K.applications.convnext.preprocess_input(X.astype('float32'))
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p

if __name__ == "__main__":
    print("Loading CIFAR-10 data.....")
    # Load the CIFAR-10 dataset
    (X_train, Y_train), (X_test, Y_test) = K.datasets.cifar10.load_data()
    X_train, Y_train = preprocess_data(X_train, Y_train)
    X_test, Y_test = preprocess_data(X_test, Y_test)
    print("Data loaded and preprocessed.")

    # Define input shape and resize layer
    input_shape = (32, 32, 3)
    resize_shape = (224, 224)
    inputs = K.layers.Input(shape=input_shape)
    resize_layer = K.layers.Lambda(lambda image: K.backend.resize_images(image, resize_shape[0] // input_shape[0], 
                                                                        resize_shape[1] // input_shape[1], 
                                                                        "channels_last"))(inputs)
    
    print("Loading pre-trained ConvNeXtTiny model...")
    # Load the ConvNeXtTiny model with pre-trained weights
    base_model = K.applications.ConvNeXtTiny(weights='imagenet', include_top=False, input_tensor=resize_layer)
    print("Model loaded.")

    print("Freezing base model layers...")
    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False
    print("Base model layers frozen.")

    print("Computing features for the training set...")
    # Extract features with the frozen base model for training set
    X_train_features = base_model.predict(X_train)
    print("Features computed for the training set.")

    print("Computing features for the validation set...")
    # Extract features with the frozen base model for validation set
    X_test_features = base_model.predict(X_test)
    print("Features computed for the validation set.")

    print("Building new model...")
    # Add custom top layers for CIFAR-10 classification
    x = K.layers.GlobalAveragePooling2D()(base_model.output)
    x = K.layers.Dense(256, activation='relu')(x)
    x = K.layers.Dropout(0.5)(x)
    outputs = K.layers.Dense(10, activation='softmax')(x)

    # Create the model
    model = K.models.Model(inputs=inputs, outputs=outputs)
    print("Model built.")

    print("Compiling the model...")
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print("Model compiled.")

    print("Training the model...")
    # Train the model
    model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, batch_size=128)
    print("Model trained.")

    print("Saving the model to cifar10.h5...")
    # Save the model
    with K.utils.custom_object_scope({'LayerScale': K.layers.Layer}):
        model.save('cifar10.h5')
    print("Model saved successfully.")

