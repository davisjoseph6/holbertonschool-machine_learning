# TensorFlow Neural Network Project

This project demonstrates the construction, training, evaluation, and saving of a neural network classifier using TensorFlow (compat.v1). Each script in this directory implements a specific aspect of the neural network pipeline, from creating placeholders to training and evaluating the model.

---

## Directory Overview

### Neural Network Components
1. **`0-create_placeholders.py`**
   - Creates placeholders for input features (`x`) and labels (`y`) of the neural network.
   - Example:  
     ```python
     x, y = create_placeholders(nx, classes)
     ```

2. **`1-create_layer.py`**
   - Implements a function to create a single neural network layer with a specified number of nodes and activation function.
   - Example:  
     ```python
     layer = create_layer(prev_layer, n, activation)
     ```

3. **`2-forward_prop.py`**
   - Builds the forward propagation graph for the neural network by stacking layers.
   - Example:  
     ```python
     y_pred = forward_prop(x, layer_sizes, activations)
     ```

4. **`3-calculate_accuracy.py`**
   - Calculates the accuracy of predictions by comparing predicted and true labels.
   - Example:  
     ```python
     accuracy = calculate_accuracy(y, y_pred)
     ```

5. **`4-calculate_loss.py`**
   - Computes the softmax cross-entropy loss for a prediction.
   - Example:  
     ```python
     loss = calculate_loss(y, y_pred)
     ```

6. **`5-create_train_op.py`**
   - Creates a training operation using gradient descent optimization.
   - Example:  
     ```python
     train_op = create_train_op(loss, alpha)
     ```

---

### Training and Evaluation
7. **`6-train.py`**
   - Builds, trains, and saves the neural network model.
   - Key features:
     - Iterative training with cost and accuracy monitoring.
     - Saves the trained model to a specified path.
   - Example usage:  
     ```python
     save_path = train(X_train, Y_train, X_valid, Y_valid, layer_sizes,
                       activations, alpha, iterations)
     ```

8. **`7-evaluate.py`**
   - Loads a saved model and evaluates its performance on a given dataset.
   - Returns predictions, accuracy, and loss.
   - Example usage:  
     ```python
     prediction, accuracy, loss = evaluate(X_test, Y_test, save_path)
     ```

---

## Additional Files
- **`model.ckpt.*`**: Files generated during the training process to store the model's weights and metadata.
- **`checkpoint`**: TensorFlow checkpoint file.
- **`output.log` / `error.log`**: Logs generated during script execution.
- **`README.md`**: Documentation for the project.

---

## Usage
### Running Scripts
- Execute individual scripts to test specific components.
- Example:  
  ```bash
  python3 0-create_placeholders.py
  ```

### Training the Model
- Train a neural network using 6-train.py with specified hyperparameters:
```python
save_path = train(X_train, Y_train, X_valid, Y_valid,
                  layer_sizes=[128, 64, 32], 
                  activations=[tf.nn.relu, tf.nn.relu, tf.nn.softmax],
                  alpha=0.01, iterations=1000)
```

- Evaluating the Model
Evaluate a saved model with `7-evaluate.py`:
```python
prediction, accuracy, loss = evaluate(X_test, Y_test, "/tmp/model.ckpt")
```

## Requirements
- Python 3.x
- TensorFlow 1.x (tensorflow.compat.v1 module)
- NumPy

## Author
Davis Joseph
