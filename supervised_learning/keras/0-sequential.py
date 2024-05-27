import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


def build_model(nx, layers_list, activations, lambtha, keep_prob):
    model = models.Sequential()
    for i in range(len(layers_list)):
        if i == 0:
            model.add(layers.Dense(layers_list[i], input_shape=(nx,), activation=activations[i],
                kernel_regularizer=regularizers.l2(lambtha)))
        else:
            model.add(layers.Dense(layers_list[i], activation=activations[i],
                kernel_regularizer=regularizers.l2(lambtha)))
        if i < len(layers_list) - 1:  # Don't add dropput after the last layer
            model.add(layers.Dropout(1 - keep_prob))
    return model
