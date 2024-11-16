# Decision Tree Project

This project is part of the Holberton School Machine Learning curriculum and focuses on supervised learning using decision trees, random forests, and isolation forests. The directory contains various Python scripts implementing these models along with supporting methods and utilities.

## Directory Overview

### Modules and Implementations

1. **`0-build_decision_tree.py` to `7-build_decision_tree.py`**
   - Various implementations of decision tree construction, covering features such as:
     - Node representation
     - Splitting criteria
     - Recursion for tree building
     - Leaf handling and bounds updating
     - Prediction functionality

2. **`8-build_decision_tree.py`**
   - Comprehensive decision tree implementation using:
     - Node and leaf classes
     - Gini impurity and random splitting criteria
     - Depth and node count calculations
     - Recursive tree-building with termination conditions
     - Prediction and accuracy functions

3. **`9-random_forest.py`**
   - Implementation of a Random Forest using multiple decision trees.
   - Features include:
     - Training and prediction across multiple decision trees
     - Calculation of mode-based predictions
     - Support for configurable parameters like `n_trees`, `max_depth`, and `min_pop`.

4. **`10-isolation_tree.py`**
   - Implementation of an Isolation Tree for anomaly detection.
   - Highlights include:
     - Random splitting of features
     - Depth-based leaf calculation
     - Prediction using isolation principles

5. **`11-isolation_forest.py`**
   - Isolation Forest implementation using multiple isolation trees.
   - Includes:
     - Aggregation of predictions from isolation trees
     - Depth-based suspect detection for identifying anomalies

---

## Features and Capabilities

### Decision Tree
- Recursive implementation for binary tree construction.
- Support for both random and Gini impurity-based split criteria.
- Prediction and accuracy measurement functions.

### Random Forest
- Ensemble learning approach with configurable tree count and depth.
- Mode-based prediction aggregation for enhanced accuracy.
- Node and leaf statistics tracking.

### Isolation Tree & Forest
- Depth-based anomaly detection.
- Aggregation of isolation tree predictions for robust outlier identification.
- Functionality to identify "suspects" (rows with shallow depth).

---

## How to Use

### Requirements
- Python 3.x
- NumPy

### Execution
- Run the individual scripts for testing and demonstration of specific functionalities.
- Example usage for `Random_Forest`:
  ```python
  from 9-random_forest import Random_Forest
  rf = Random_Forest(n_trees=10, max_depth=5)
  rf.fit(train_features, train_labels)
  predictions = rf.predict(test_features)
  print("Accuracy:", rf.accuracy(test_features, test_labels))
  ```

### Example usage for Isolation_Random_Forest:
```python
from 11-isolation_forest import Isolation_Random_Forest
irf = Isolation_Random_Forest(n_trees=50, max_depth=8)
irf.fit(dataset)
suspects, depths = irf.suspects(dataset, n_suspects=10)
print("Top 10 anomalies:", suspects)
```

### Outputs
- Decision tree visualizations through string representations.
- Depth, node, and accuracy metrics during training.
- Anomaly detection results with suspect identification.

### Authors
Davis Joseph
