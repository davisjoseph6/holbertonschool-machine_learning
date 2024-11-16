# Supervised Learning - Classification

This project is part of the Holberton School Machine Learning curriculum and focuses on supervised learning techniques for classification tasks using neural networks. The repository implements binary classification with single-layer, multi-layer, and deep neural networks.

---

## Project Overview

### 1. **Neuron Class**
- **File:** `7-neuron.py`
- Implements a single neuron for binary classification.
- **Features:**
  - Forward propagation using sigmoid activation.
  - Cost calculation with logistic regression.
  - Gradient descent for parameter updates.
  - Training function with visualization support.

### 2. **Neural Network Class**
- **File:** `15-neural_network.py`
- Implements a neural network with one hidden layer for binary classification.
- **Features:**
  - Two-layer network (input-hidden-output).
  - Forward propagation and backpropagation.
  - Training with gradient descent and cost visualization.

### 3. **Deep Neural Network Class**
- **File:** `28-deep_neural_network.py`
- Implements a deep neural network with customizable layers.
- **Features:**
  - Supports arbitrary layer architectures.
  - He-initialization for weights.
  - Multiple activation functions: Sigmoid, Tanh.
  - Gradient descent with backpropagation.
  - Save and load functionality using `pickle`.

### 4. **One-Hot Encoding and Decoding**
- **Files:** 
  - `24-one_hot_encode.py`
  - `25-one_hot_decode.py`
- Functions to convert between numeric labels and one-hot matrices for multi-class classification.

### 5. **Visualization**
- **Files:** 
  - `show_data.py`
  - `show_multi_data.py`
- Scripts to visualize data and multi-class results.

---

## Key Concepts

- **Activation Functions:** Sigmoid, Tanh, and Softmax used for various network layers.
- **Loss Function:** Logistic regression cost function for binary classification.
- **Gradient Descent:** Updates weights and biases by minimizing cost.
- **One-Hot Encoding:** Prepares data for multi-class classification tasks.

---

## Usage

1. **Training a Model**
   - Use the `train` method in the `Neuron`, `NeuralNetwork`, or `DeepNeuralNetwork` classes.
   - Enable `verbose` or `graph` to monitor training progress.

2. **Saving and Loading Models**
   - Use the `save` and `load` methods in the `DeepNeuralNetwork` class.

3. **One-Hot Operations**
   - Encode labels with `one_hot_encode`.
   - Decode predictions with `one_hot_decode`.

---

## Dependencies

- Python 3.x
- NumPy
- Matplotlib

---

## Resources

- **LinkedIn Article:** [Understanding Activation Functions in Neural Networks](https://www.linkedin.com/pulse/understanding-activation-functions-neural-networks-guide-davis-joseph-aswpe/)  
  Explains the importance of activation functions and their role in deep learning.

---

## Authors

- **Davis Joseph** - [LinkedIn](https://www.linkedin.com/in/davis-joseph-aswpe/)
- Holberton School Machine Learning Cohort

