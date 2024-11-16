# Convolutional Neural Networks (CNNs) in Machine Learning

This project focuses on building, training, and implementing Convolutional Neural Networks (CNNs) for image data. Key operations such as convolution, pooling, and backpropagation are implemented from scratch using NumPy. Additionally, the LeNet-5 architecture is built using TensorFlow and Keras.

---

## Directory Overview

### Convolution Operations
1. **`0-conv_forward.py`**
   - Implements forward propagation over a convolutional layer of a neural network.

2. **`2-conv_backward.py`**
   - Implements backpropagation over a convolutional layer to compute gradients with respect to inputs, weights, and biases.

---

### Pooling Operations
3. **`1-pool_forward.py`**
   - Implements forward propagation over a pooling layer using max or average pooling.

4. **`3-pool_backward.py`**
   - Implements backpropagation for pooling layers to compute gradients with respect to inputs.

---

### LeNet-5 Architecture
5. **`5-lenet5.py`**
   - Builds a modified version of the LeNet-5 architecture using Keras.
   - The architecture includes convolutional, pooling, and fully connected layers, optimized with Adam and categorical crossentropy.

---

## How to Use

### Forward Propagation
- Use `0-conv_forward.py` for convolutional layers and `1-pool_forward.py` for pooling layers to process inputs through a CNN.

### Backward Propagation
- Use `2-conv_backward.py` and `3-pool_backward.py` for backpropagation through convolutional and pooling layers, respectively.

### Building a CNN
- Utilize `5-lenet5.py` to build a LeNet-5 CNN model with Keras. This model is suitable for image classification tasks like MNIST.

---

## Features

- **Custom Implementations**: Convolutional and pooling layers are implemented from scratch using NumPy.
- **LeNet-5 Architecture**: A ready-to-use CNN model for image recognition.
- **Backward Propagation**: Compute gradients to update parameters during training.

---

## Applications
- Image recognition and classification.
- Deep learning model design and experimentation.
- Understanding CNN architectures and their components.

---

## Requirements
- Python 3.x
- NumPy
- TensorFlow 2.x

---

## Author
- Davis Joseph ([LinkedIn](https://www.linkedin.com/in/davis-joseph/))

