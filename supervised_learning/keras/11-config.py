#!/usr/bin/env python3
"""
Saves a model's configuration in JSON format
Loads a model with a specific configuration
"""

import tensorflow.keras as K


def save_config(network, filename):
    """
    Saves a model's configuration in JSON format.
    """
    config = network.to_json()
    with open(filename, 'w') as json_file:
        json_file.write(config)


def load_config(filename):
    """
    Loads a model with a specific configuration.
    """
    with open(filename, 'r') as json_file:
        config = json_file.read()
    return K.models.model_from_json(config)
