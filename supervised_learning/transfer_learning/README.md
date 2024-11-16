# Transfer Learning for CIFAR-10 Classification

This project demonstrates the use of transfer learning for image classification tasks. Specifically, a pre-trained DenseNet-121 model is adapted to classify the CIFAR-10 dataset. Transfer learning allows leveraging pre-trained models on large datasets (e.g., ImageNet) to improve performance on smaller, domain-specific datasets.

---

## Directory Overview

### Scripts
1. **`0-transfer.py`**
   - Implements a transfer learning pipeline using DenseNet-121 pre-trained on ImageNet.
   - Includes data preprocessing, model building, training, and saving.

### Pre-trained Models
- **`VGG-16/`**: Directory to experiment with VGG-16 for transfer learning.
- **`ConvNeXtTiny/`** and **`ConvNeXtSmall/`**: Explore ConvNeXt models for advanced transfer learning.

---

## Key Features

### Transfer Learning with DenseNet-121
- **Feature Extraction**: The pre-trained DenseNet-121 model is used as a feature extractor with `include_top=False`, meaning the classification layers are excluded.
- **Custom Classification Head**: A custom head is added, comprising:
  - Global average pooling layer
  - Dense layer with 256 units and ReLU activation
  - Dense output layer with 10 units and softmax activation for CIFAR-10 classes

### CIFAR-10 Dataset
- **Preprocessing**: Images are resized to match the DenseNet-121 input requirements (224x224x3), and labels are one-hot encoded.
- **Training and Evaluation**: The model is trained for 10 epochs with the Adam optimizer and categorical crossentropy loss, and performance is validated on the test set.

---

## How to Use

1. **Run the Script**
   - Execute `0-transfer.py` to train and save the model:
     ```bash
     python3 0-transfer.py
     ```

2. **Pre-trained Models**
   - Experiment with other architectures (e.g., VGG-16, ConvNeXt) by modifying the base model in `0-transfer.py`.

3. **Saved Model**
   - The trained model is saved as `cifar10.h5` and can be loaded for inference or further training:
     ```python
     model = K.models.load_model('cifar10.h5')
     ```

---

## Applications
- Transfer learning for small datasets
- Image classification tasks with limited data
- Experimenting with various pre-trained models for feature extraction

---

## Requirements
- Python 3.x
- TensorFlow 2.x
- NumPy

---

## References
- *"Transfer Learning for CIFAR-10 Classification Using VGG-16"*  
  [LinkedIn Article](https://www.linkedin.com/pulse/transfer-learning-cifar-10-classification-using-vgg16-davis-joseph-kdive/)

---

## Author
- Davis Joseph ([LinkedIn]([https://www.linkedin.com/in/davis-joseph/](https://www.linkedin.com/in/davisjoseph767/)))

