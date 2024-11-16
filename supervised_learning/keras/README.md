# Keras Project

This project is part of the Holberton School Machine Learning curriculum and focuses on building, training, saving, and testing neural networks using the Keras library.

---

## Directory Overview

### Neural Network Construction
1. **`0-sequential.py`**
   - Builds a neural network using the Keras Sequential API.
   - Supports L2 regularization and dropout.

2. **`1-input.py`**
   - Constructs a neural network using the Keras Functional API.
   - Allows flexibility in defining complex architectures.

---

### Model Optimization
3. **`2-optimize.py`**
   - Sets up the Adam optimizer with categorical cross-entropy loss and accuracy metrics.

4. **`3-one_hot.py`**
   - Converts label vectors into one-hot encoded matrices.

---

### Model Training
5. **`8-train.py`**
   - Implements mini-batch gradient descent with:
     - Optional validation data
     - Early stopping
     - Learning rate decay
     - Model checkpointing

---

### Model Persistence
6. **`9-model.py`**
   - Provides methods to save and load entire models.

7. **`10-weights.py`**
   - Enables saving and loading of model weights independently.

8. **`11-config.py`**
   - Saves model configurations in JSON format and reconstructs models from these configurations.

---

### Model Evaluation
9. **`12-test.py`**
   - Evaluates a model's performance on test data.

10. **`13-predict.py`**
    - Makes predictions using a trained model.

---

## How to Use

### Building Models
- Use `0-sequential.py` or `1-input.py` to define a neural network depending on your preferred Keras API.

### Optimizing Models
- Optimize models with `2-optimize.py` using Adam and categorical cross-entropy.

### Training
- Train the model with `8-train.py` and add callbacks for early stopping, learning rate decay, and model checkpointing as needed.

### Saving and Loading
- Save models and weights using `9-model.py` or `10-weights.py`.
- Use `11-config.py` to save and reload model configurations.

### Evaluation and Prediction
- Evaluate model accuracy using `12-test.py`.
- Make predictions with `13-predict.py`.

---

## Requirements
- Python 3.x
- TensorFlow/Keras 2.x

---

## Authors
- Holberton School Machine Learning Cohort

