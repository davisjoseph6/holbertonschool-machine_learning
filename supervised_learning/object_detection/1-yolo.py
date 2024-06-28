#!/usr/bin/env python3

import tensorflow.keras as K
import numpy as np


class Yolo:
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Initialize the Yolo class with the specified parameters
        """
        self.model = K.models.load_model(model_path)
        self.class_names = self._load_class_names(classes_path)
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def _load_class_names(self, classes_path):
        """
        Load class names from a file
        """
        with open(classes_path, 'r') as file:
            class_names = file.read().strip().split('\n')
        return class_names

    def process_outputs(self, outputs, image_size):
        """
        Process the outputs from the Darknet model
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        for i, output in enumerate(outputs):
            grid_height, grid_width, anchor_boxes, _ = output.shape
            anchors = self.anchors[i]
            box_xy = K.activations.sigmoid(output[..., :2])
            box_wh = np.exp(output[..., 2:4]) * anchors
            box_confidence = K.activations.sigmoid(output[..., 4:5])
            box_class_probs_array = K.activations.sigmoid(output[..., 5:])

            col = np.tile(np.arange(0, grid_width), grid_height).reshape(-1, grid_width)
            row = np.tile(np.arange(0, grid_height), grid_width).reshape(-1, grid_height).T
            col = col.reshape(grid_height, grid_width, 1, 1).repeat(3, axis=-2)
            row = row.reshape(grid_height, grid_width, 1, 1).repeat(3, axis=-2)
            grid = np.concatenate((col, row), axis=-1)

            box_xy += grid
            box_xy /= (grid_width, grid_height)
            box_wh /= image_size
            
            box_xy -= (box_wh / 2)
            box = np.concatenate((box_xy, box_xy + box_wh), axis=-1)

            boxes.append(box)
            box_confidences.append(box_confidence.numpy())
            box_class_probs.append(box_class_probs_array.numpy())

        return boxes, box_confidences, box_class_probs

