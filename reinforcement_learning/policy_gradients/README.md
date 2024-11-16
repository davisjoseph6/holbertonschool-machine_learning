# Reinforcement Learning: Policy Gradients

This project demonstrates the implementation of a basic reinforcement learning (RL) algorithm using policy gradients. The project consists of several Python scripts that define the policy gradient method, training loop, and utility functions for training a model on a given environment.

## Directory Overview

### Policy Gradient
1. **`policy_gradient.py`**
   - Contains the `policy_gradient` function that computes the action and its gradient using policy gradient methods for reinforcement learning. It includes:
     - **`policy(matrix, weight)`**: Computes the policy using a softmax function over the state and weight matrix.
     - **`policy_gradient(state, weight)`**: Computes the Monte Carlo policy gradient, selects actions based on computed probabilities, and calculates the gradient.

### Training Loop
2. **`train.py`**
   - Implements the full training process using the policy gradient method. The function `train()` performs training over a set number of episodes, updating the policy's weights using the gradients computed at each step.

   **Parameters:**
   - `env`: The environment to train the model on.
   - `nb_episodes`: Number of episodes to train the model.
   - `alpha`: Learning rate (default: 0.000045).
   - `gamma`: Discount factor for future rewards (default: 0.98).
   - `show_result`: Boolean flag to display the environment (default: False).

   **Process:**
   - For each episode, the environment is reset and actions are sampled from the policy.
   - Rewards are collected and used to calculate the gradient, which is then used to update the policy's weights.

---

# Supervised Learning: Error Analysis

This project contains scripts for calculating various error metrics, including confusion matrix, sensitivity, specificity, precision, and F1 score.

## Directory Overview

### Error Metrics Calculation
1. **`0-create_confusion.py`**
   - Creates a confusion matrix from true labels and predicted logits.
   - **Function: `create_confusion_matrix(labels, logits)`**: 
     - **`labels`**: One-hot encoded true labels.
     - **`logits`**: Predicted probabilities or logits.

2. **`1-sensitivity.py`**
   - Calculates the sensitivity for each class in a confusion matrix.
   - **Sensitivity** is calculated as:
     \[
     \text{Sensitivity} = \frac{TP}{TP + FN}
     \]
     where \(TP\) is true positives and \(FN\) is false negatives.

3. **Other scripts**:
   - **`2-main.py`**: May calculate precision.
   - **`3-main.py`**: Likely calculates specificity.
   - **`4-f1_score.py`**: Computes the F1 score.
   - **`5-error_handling`**: Handles error scenarios, likely dealing with invalid input or edge cases.
   - **`6-compare_and_contrast`**: Compares different error metrics.

---

## Requirements
- Python 3.x
- NumPy

---

## Usage
To use the provided functions, import the respective modules and pass in your data (e.g., labels and logits for error metrics, or environment for RL training).

Example:
```python
from policy_gradient import policy_gradient

state = np.array([1, 2, 3, 4])
weight = np.random.rand(4, 2)
action, gradient = policy_gradient(state, weight)
print(f"Action: {action}, Gradient: {gradient}")
```

For training, use the train.py script by specifying your environment and other parameters:

```python
from train import train

env = some_reinforcement_learning_environment()
scores = train(env, nb_episodes=1000, alpha=0.001, gamma=0.99)
```

## Authors
- Davis Joseph
