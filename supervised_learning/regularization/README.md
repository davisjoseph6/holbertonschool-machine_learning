# Regularization Techniques in Neural Networks

This project focuses on implementing various regularization techniques to improve the performance and generalization of neural networks. Techniques like L2 regularization, dropout, and early stopping are explored and implemented both in pure NumPy and TensorFlow.

---

## Directory Overview

### L2 Regularization
1. **`0-l2_reg_cost.py`**
   - Calculates the cost of a neural network with L2 regularization using NumPy.

2. **`1-l2_reg_gradient_descent.py`**
   - Updates the weights and biases of a neural network using gradient descent with L2 regularization.

3. **`2-l2_reg_cost.py`**
   - Computes the L2 regularization cost for a neural network using TensorFlow.

4. **`3-l2_reg_create_layer.py`**
   - Creates a neural network layer with L2 regularization in TensorFlow.

---

### Dropout Regularization
5. **`4-dropout_forward_prop.py`**
   - Implements forward propagation with dropout regularization using NumPy.

6. **`5-dropout_gradient_descent.py`**
   - Updates the weights of a neural network with dropout regularization using gradient descent in NumPy.

7. **`6-dropout_create_layer.py`**
   - Creates a dense layer with dropout using TensorFlow.

---

### Early Stopping
8. **`7-early_stopping.py`**
   - Implements early stopping to determine if training should halt based on performance.

---

## How to Use

### L2 Regularization
- Use **`0-l2_reg_cost.py`** and **`2-l2_reg_cost.py`** to compute the L2 regularization cost during training.
- Apply gradient descent with L2 regularization using **`1-l2_reg_gradient_descent.py`**.
- Use **`3-l2_reg_create_layer.py`** to create TensorFlow layers with L2 regularization.

### Dropout Regularization
- Perform forward propagation with dropout using **`4-dropout_forward_prop.py`**.
- Update weights with dropout regularization using **`5-dropout_gradient_descent.py`**.
- Create dropout-enabled layers in TensorFlow using **`6-dropout_create_layer.py`**.

### Early Stopping
- Use **`7-early_stopping.py`** to halt training when improvements in the cost function fall below a specified threshold.

---

## Additional Resources

For an in-depth explanation of these regularization techniques and their importance, refer to the article:  
[Enhancing Neural Networks: Exploring Regularization](https://www.linkedin.com/pulse/enhancing-neural-networks-exploring-regularization-davis-joseph-oekme/)

---

## Requirements
- Python 3.x
- NumPy
- TensorFlow 2.x

---

## Author
- Davis Joseph ([LinkedIn](https://www.linkedin.com/in/davis-joseph/))

