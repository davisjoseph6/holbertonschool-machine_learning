# YOLO Object Detection

This project implements the YOLO (You Only Look Once) object detection model using TensorFlow/Keras. It includes methods for loading pre-trained YOLO models, processing outputs, filtering predictions, and visualizing results on images.

---

## Directory Overview

### Key Files
1. **`0-yolo.py` to `7-yolo.py`**
   - Incremental implementations of the YOLO object detection pipeline, culminating in `7-yolo.py`, which contains the complete YOLO class.

2. **`detections/`**
   - Directory where processed images with bounding boxes and class predictions are saved.

### Main Scripts
- **`0-main.py` to `7-main.py`**
  - Test scripts corresponding to each YOLO implementation file.

---

## Features

### YOLO Class (`7-yolo.py`)
- **Initialization**: Loads the YOLO model and configurations, including:
  - Pre-trained model weights
  - Class names
  - Thresholds for class confidence (`class_t`) and Non-Max Suppression (`nms_t`)
  - Anchor boxes

- **Core Methods**:
  1. **`process_outputs(outputs, image_size)`**:
     - Processes model outputs to extract bounding boxes, confidence scores, and class probabilities.
  2. **`filter_boxes(boxes, box_confidences, box_class_probs)`**:
     - Filters low-confidence boxes based on a threshold.
  3. **`non_max_suppression(filtered_boxes, box_classes, box_scores)`**:
     - Applies Non-Max Suppression to eliminate overlapping boxes.
  4. **`load_images(folder_path)`**:
     - Loads images from a specified folder.
  5. **`preprocess_images(images)`**:
     - Resizes and normalizes images for YOLO model input.
  6. **`show_boxes(image, boxes, box_classes, box_scores, file_name)`**:
     - Visualizes bounding boxes and class predictions on images.
  7. **`predict(folder_path)`**:
     - Executes the complete object detection pipeline and displays results.

- **Utilities**:
  - Intersection Over Union (IoU) calculation for overlapping boxes.
  - Image loading and preprocessing for model compatibility.

---

## How to Use

1. **Set Up YOLO**:
   - Download a pre-trained YOLO model and place it in the appropriate directory.
   - Provide class names, anchors, and thresholds as required.

2. **Run Object Detection**:
   - Use `7-main.py` to load the YOLO model and run detection:
     ```bash
     python3 7-main.py
     ```

3. **Visualize Results**:
   - Processed images with bounding boxes and class predictions are displayed. Save detections by pressing **'s'** when prompted.

4. **Explore Outputs**:
   - Processed images are saved in the `detections/` directory.

---

## Applications
- Real-time object detection in images and videos.
- Automated object recognition and localization.
- Applications in surveillance, autonomous driving, and more.

---

## Requirements
- Python 3.x
- TensorFlow 2.x
- OpenCV
- NumPy

---

## References
- *"You Only Look Once: Unified, Real-Time Object Detection"* (YOLO)

---

## Author
- Davis Joseph ([LinkedIn]([https://www.linkedin.com/in/davis-joseph/](https://www.linkedin.com/in/davisjoseph767/)))

