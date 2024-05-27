#!/usr/bin/env python3

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    model = K.Sequential()
    for i in range(len(layers)):
        if i == 0:
            model.add(K.layers.Dense(layers[i], input_shape=(nx,),
                activation=activations[i], kernel_regularizer=K.regularizers.
                l2(lambtha)))
        else:
            model.add(K.layers.Dense(layers[i], activation=activations[i],
                kernel_regularizer=K.regularizers.l2(lambtha)))
        if i < len(layers) - 1:
            model.add(K.layers.Dropout(1 - keep_prob))
    return model
