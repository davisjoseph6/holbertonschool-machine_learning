#!/usr/bin/env python3
"""
Yolo class
"""
import tensorflow.keras as K
import numpy as np


class Yolo:
    """
    Yolo class
    """
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

        image_height, image_width = image_size

        for i, output in enumerate(outputs):
            grid_height, grid_width, anchor_boxes, _ = output.shape
            anchors = self.anchors[i]
            box_xy = K.activations.sigmoid(output[..., :2])
            box_wh = np.exp(output[..., 2:4]) * anchors
            box_confidence = K.activations.sigmoid(output[..., 4:5])
            box_class_probs_array = K.activations.sigmoid(output[..., 5:])

            col = np.tile(np.arange(0, grid_width),
                          grid_height).reshape(-1, grid_width)
            row = np.tile(np.arange(0, grid_height),
                          grid_width).reshape(-1, grid_height).T
            col = col.reshape(grid_height,
                              grid_width, 1, 1).repeat(anchor_boxes, axis=-2)
            row = row.reshape(grid_height,
                              grid_width, 1, 1).repeat(anchor_boxes, axis=-2)
            grid = np.concatenate((col, row), axis=-1)

            box_xy += grid
            box_xy /= (grid_width, grid_height)
            box_wh /= np.array([self.model.input.shape[1],
                                self.model.input.shape[2]])

            box_xy -= (box_wh / 2)
            box = np.concatenate((box_xy, box_xy + box_wh), axis=-1)

            # Scale boxes back to original image size
            box[..., 0:4:2] *= image_width
            box[..., 1:4:2] *= image_height

            boxes.append(box)
            box_confidences.append(box_confidence.numpy())
            box_class_probs.append(box_class_probs_array.numpy())

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filter the boxes, class predictions, and box scores to remove
        low-scoring boxes
        """
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for i in range(len(boxes)):
            box_scores_array = box_confidences[i] * box_class_probs[i]
            box_classes_array = np.argmax(box_scores_array, axis=-1)
            box_class_scores = np.max(box_scores_array, axis=-1)

            pos = np.where(box_class_scores >= self.class_t)

            filtered_boxes.append(boxes[i][pos])
            box_classes.append(box_classes_array[pos])
            box_scores.append(box_class_scores[pos])

        filtered_boxes = np.concatenate(filtered_boxes)
        box_classes = np.concatenate(box_classes)
        box_scores = np.concatenate(box_scores)

        return filtered_boxes, box_classes, box_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Apply non-max suppression to filter the boxes
        """
        indices = np.lexsort((-box_scores, box_classes))
        filtered_boxes = filtered_boxes[indices]
        box_scores = box_scores[indices]
        box_classes = box_classes[indices]

        unique_classes = np.unique(box_classes)
        nms_boxes = []
        nms_classes = []
        nms_scores = []

        for cls in unique_classes:
            cls_indices = np.where(box_classes == cls)[0]
            cls_boxes = filtered_boxes[cls_indices]
            cls_scores = box_scores[cls_indices]

            while len(cls_boxes) > 0:
                max_score_index = np.argmax(cls_scores)
                nms_boxes.append(cls_boxes[max_score_index])
                nms_classes.append(cls)
                nms_scores.append(cls_scores[max_score_index])

                if len(cls_boxes) == 1:
                    break

                cls_boxes = np.delete(cls_boxes, max_score_index, axis=0)
                cls_scores = np.delete(cls_scores, max_score_index)

                ious = self._iou(nms_boxes[-1], cls_boxes)
                iou_indices = np.where(ious <= self.nms_t)[0]

                cls_boxes = cls_boxes[iou_indices]
                cls_scores = cls_scores[iou_indices]

        return (np.array(nms_boxes),
                np.array(nms_classes),
                np.array(nms_scores))

    def _iou(self, box1, box2):
        """
        Calculate the Intersection Over Union (IOU) of two bounding boxes
        """
        x1 = np.maximum(box1[0], box2[:, 0])
        y1 = np.maximum(box1[1], box2[:, 1])
        x2 = np.minimum(box1[2], box2[:, 2])
        y2 = np.minimum(box1[3], box2[:, 3])

        inter_area = np.maximum((x2 - x1), 0) * np.maximum((y2 - y1), 0)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area
