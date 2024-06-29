#!/usr/bin/env python3
"""
    Initialize Yolo
"""
import cv2
import os
import tensorflow as tf
import numpy as np


class Yolo:
    """
        Class Yolo uses the Yolo v3 algorithm to perform object detection
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
            Class constructor of Yolo class

            :param model_path: path where Darknet Keras model is stored
            :param classes_path:path where list of class names,
                in order of index
            :param class_t: float, box score threshold
                for initial filtering step
            :param nms_t: float, IOU threshold for non-max suppression
            :param anchors: ndarray, shape(outputs, anchor_boxes, 2)
                all anchor boxes
                outputs: number of outputs (prediction) made by Darknet model
                anchor_boxes: number of anchor boxes used for each prediction
                2: [anchor_box_width, anchor_box_height]

        """
        self.model = tf.keras.models.load_model(model_path)
        self.class_names = []
        with open(classes_path, 'r') as f:
            for line in f:
                line = line.strip()
                self.class_names.append(line)
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """
            Function to process outputs

        :param outputs: list of ndarray, predictions from a single image
                each output,
                shape(grid_height, grid_width, anchor_boxes, 4+1+classes)
                grid_height, grid_width: height and width of grid
                 used for the output
                anchor_boxes: number of anchor boxes used
                4 => (t_x, t_y, t_w, t_h)
                1 => box_confidence
                classes => classes probabilities for all classes
        :param image_size: ndarray,
               image's original size [image_height, image_width]

        :return: tuple (boxes, box_confidences, box_class_probs):
                boxes: list of ndarrays,
                       shape(grid_height, grid_width, anchor_boxes, 4)
                        processed boundary boxes for each output
                        4 => (x1,y1, x2, y2)
                boxe_confidences: list ndarray,
                    shape(grid_height, grid_width, anchor_boxes, 1)
                    boxe confidences for each output
                box_class_probs: list ndarray,
                    shape(grid_height, grid_width, anchor_boxes, classes)
                    box's class probabilities for each output
        """
        # extract image size
        image_height, image_height = image_size

        boxes = []
        box_confidences = []
        box_class_probs = []

        # process for each output
        for idx, output in enumerate(outputs):
            # extract height, width, number of anchor box for current output
            grid_height, grid_width, nbr_anchor, _ = output.shape

            # extract coordinate of output NN
            t_x = output[:, :, :, 0]
            t_y = output[:, :, :, 1]
            t_w = output[:, :, :, 2]
            t_h = output[:, :, :, 3]

            # grid coordinate
            grid_x, grid_y = np.meshgrid(np.arange(grid_width),
                                         np.arange(grid_height))

            # Repeat grid coordinate for each anchor box
            grid_x = np.expand_dims(grid_x, axis=-1)
            grid_y = np.expand_dims(grid_y, axis=-1)

            # extract anchor_box_width, anchor_box_height
            p_w = self.anchors[idx, :, 0]
            p_h = self.anchors[idx, :, 1]

            # size image
            image_height, image_width = image_size

            # sigmoid : grid scale (value between 0 and 1)
            # + c_x or c_y : coordinate of cells in the grid
            b_x = ((1.0 / (1.0 + np.exp(-t_x))) + grid_x) / grid_width
            b_y = ((1.0 / (1.0 + np.exp(-t_y))) + grid_y) / grid_height
            # exp for predicted height and width
            b_w = p_w * np.exp(t_w)
            b_w /= self.model.input.shape[1].value
            b_h = p_h * np.exp(t_h)
            b_h /= self.model.input.shape[2].value

            # conv in pixel : absolute coordinate
            x1 = (b_x - b_w / 2) * image_width
            y1 = (b_y - b_h / 2) * image_height
            x2 = (b_w / 2 + b_x) * image_width
            y2 = (b_h / 2 + b_y) * image_height

            # Update box array with box coordinates and dimensions
            box = np.zeros((grid_height, grid_width, nbr_anchor, 4))
            box[:, :, :, 0] = x1
            box[:, :, :, 1] = y1
            box[:, :, :, 2] = x2
            box[:, :, :, 3] = y2
            boxes.append(box)

            confidences = output[:, :, :, 4:5]
            sigmoid_confidence = 1 / (1 + np.exp(-confidences))
            class_probs = output[:, :, :, 5:]
            sigmoid_class_probs = 1 / (1 + np.exp(-class_probs))

            box_confidences.append(sigmoid_confidence)
            box_class_probs.append(sigmoid_class_probs)

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
            Public method to filter boxes of preprocess method

        :param boxes: list of ndarray,
              shape(grid_height, grid_width, anchor_boxes, 4)
             processed boundary boxes for each output
        :param box_confidences: list of ndarray,
            shape(grid_height, grid_width, anchor_boxes, 1)
            processed box confidences for each output
        :param box_class_probs: list of ndarray,
            shape(grid_height, grid_width, anchor_boxes, classes)
            processed box class probabilities for each output
        :return: tuple of (filtered_boxes, box_classes, box_scores)
            - filtered_boxes: ndarray, shape(?, 4)
                containing all of the filtered bounding boxes
            - box_classes: ndarray, shape(?,)
                 class number that each box in filtered_boxes predicts
            - box_scores: ndarray,  shape(?)
                box scores for each box in filtered_boxes
        """

        # initialize with 4 col to be wompatible with mask
        filtered_boxes = np.empty((0, 4))
        box_classes = np.empty((0,), dtype=int)
        box_scores = np.empty(0, dtype=int)

        for i in range(len(boxes)):
            # box score
            box_score = np.multiply(box_confidences[i], box_class_probs[i])

            # find box_classes with max box_scores
            box_classes_i = np.argmax(box_score, axis=-1)
            box_class_score = np.max(box_score, axis=-1)

            # create filtering mask
            filtering_mask = box_class_score >= self.class_t

            # apply mask and concatenate boxes
            filtered_boxes = np.concatenate((filtered_boxes,
                                             boxes[i][filtering_mask]), axis=0)
            box_classes = (
                np.concatenate((box_classes,
                                box_classes_i[filtering_mask]),
                               axis=0))
            box_scores = np.concatenate((box_scores,
                                         box_class_score[filtering_mask]),
                                        axis=0)

        return filtered_boxes, box_classes, box_scores

    def iou(self, box1, box2):
        """
            Execute Intersection over Union (IoU) between 2 box

            :param box1: coordinate box1
            :param box2: coordinate box2

            :return: float, the IoU value between the two bounding boxes
        """
        b1x1, b1y1, b1x2, b1y2 = tuple(box1)
        b2x1, b2y1, b2x2, b2y2 = tuple(box2)

        # calculate intersection of box1 and box2
        x1 = np.maximum(b1x1, b2x1)
        y1 = np.maximum(b1y1, b2y1)
        x2 = np.minimum(b1x2, b2x2)
        y2 = np.minimum(b1y2, b2y2)

        # calculate inter and  Union(A,B) = A + B - Inter(A,B)
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area1 = (b1x2 - b1x1) * (b1y2 - b1y1)
        area2 = (b2x2 - b2x1) * (b2y2 - b2y1)
        union = area1 + area2 - intersection

        # compute score IoU
        result = intersection / union

        return result

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
            method to apply Non-max Suppression
            (suppress overlapping box)

            :param filtered_boxes: ndarray, shape(?,4)
                    all filtered bounding boxes
            :param box_classes: ndarray, shape(?,)
                    class number for class that filtered_boxes predicts
            :param box_scores: ndarray, shape(?)
                box scores for each box in filtered_boxes

            :return: tuple (box_predictions, predicted_box_classes,
             predicted_box_scores)
                - box_predictions : ndarray, shape(?,4)
                    all predicted bounding boxes ordered by class and box score
                - predicted_box_classes: ndarray, shape(?,)
                    class number for box_predictions ordered by class and box
                    score
                - predicted_box_scores: ndarray, shape(?)
                    box scores for box_predictions ordered by class and box
                    score
        """
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        # Iterate over each unique class
        unique_classes = np.unique(box_classes)
        for cls in unique_classes:

            # Get indices of boxes (idx line) belonging to the current class
            class_indices = np.where(box_classes == cls)[0]

            # boxes and scores for the current class
            class_boxes = filtered_boxes[class_indices]
            class_scores = box_scores[class_indices]

            # while boxes remain in the class_boxes list
            while len(class_boxes) > 0:
                # find the index of highest scoring box for the class
                max_score_index = np.argmax(class_scores)

                # add box, class, and score to output lists
                box_predictions.append(class_boxes[max_score_index])
                predicted_box_classes.append(cls)
                predicted_box_scores.append(class_scores[max_score_index])

                # get iou scores for max box and each box in class_boxes
                ious = np.array([self.iou(class_boxes[max_score_index],
                                          box) for box in class_boxes])

                # find all boxes with an IoU greater than the threshold
                # Use [0] to get the array directly
                above_threshold = np.where(ious > self.nms_t)[0]

                # remove boxes and their scores that fell above the threshold
                if len(class_boxes) > 0:
                    class_boxes = np.delete(class_boxes, above_threshold,
                                            axis=0)
                    class_scores = np.delete(class_scores, above_threshold)

        # Convert output lists to numpy arrays
        box_predictions = np.array(box_predictions)
        predicted_box_classes = np.array(predicted_box_classes)
        predicted_box_scores = np.array(predicted_box_scores)

        return box_predictions, predicted_box_classes, predicted_box_scores

    @staticmethod
    def load_images(folder_path):
        """
            method to load images

            :param folder_path: string, path the folder holding
                all the images to load

            :return: tuple (images, image_paths)
                images : list of images as ndarray
                image_paths: list of paths to the
                individual images in images
        """
        images = []
        images_paths = []
        for filename in os.listdir(folder_path):
            # check format of image
            if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                # construct path
                images_path = os.path.join(folder_path, filename)
                # extract image with openCV
                image = cv2.imread(images_path)
                if image is not None:
                    images.append(image)
                    images_paths.append(images_path)

        return images, images_paths

    def preprocess_images(self, images):
        """
            method to preprocess images
            resize with intercubic interpolation
            rescale image in range [0, 1]

            :param images: list of images as ndarray

            :return: tuple of (pimages, image_shapes)
                pimages: ndarray, shape(ni,input_h,input_w,3)
                    ni: number of images that were preprocessed
                    input_h: input height for Darknet model
                    input_w: input width for Darknet model
                    3: number of channels
                image_shapes: ndarray, shape(ni,2)
                    original height and width
                    2 => (image_height, image_width)
        """
        pimages = []
        image_shapes = []

        for image in images:
            # extract height, width, channel from image
            h, w, c = image.shape
            image_shapes.append([h, w])

            # resize image
            input_h = self.model.input.shape[1]
            input_w = self.model.input.shape[2]
            resized_img = cv2.resize(image,
                                     dsize=(
                                         input_h,
                                         input_w),
                                     interpolation=cv2.INTER_CUBIC)

            # rescale
            resized_img = resized_img / 255.0

            pimages.append(resized_img)

        # conversion in ndarray
        pimages = np.array(pimages)
        image_shapes = np.array(image_shapes)

        return pimages, image_shapes
