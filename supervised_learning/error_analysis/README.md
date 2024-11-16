# Error Analysis in Machine Learning

This project focuses on understanding and analyzing the performance of machine learning models using error analysis techniques. It involves constructing confusion matrices and deriving important performance metrics such as sensitivity, precision, specificity, and F1 score for evaluating classification models.

---

## Directory Overview

### Confusion Matrix
1. **`0-create_confusion.py`**
   - Creates a confusion matrix to summarize the performance of a classification model.

### Performance Metrics
2. **`1-sensitivity.py`**
   - Calculates the sensitivity (recall) for each class in a confusion matrix.  
     \[
     \text{Sensitivity} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}
     \]

3. **`2-precision.py`**
   - Calculates the precision for each class in a confusion matrix.  
     \[
     \text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}
     \]

4. **`3-specificity.py`**
   - Calculates the specificity for each class in a confusion matrix.  
     \[
     \text{Specificity} = \frac{\text{True Negatives}}{\text{True Negatives} + \text{False Positives}}
     \]

5. **`4-f1_score.py`**
   - Computes the F1 score for each class in a confusion matrix.  
     \[
     \text{F1 Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
     \]

---

## Additional Files
- **`confusion.npz`**: A saved NumPy file containing an example confusion matrix for testing.
- **`5-error_handling`**: Example test data for error-handling scenarios.

---

## How to Use

1. **Generate a Confusion Matrix**
   - Use `0-create_confusion.py` to create a confusion matrix from one-hot encoded labels and predictions.

2. **Evaluate Performance Metrics**
   - Use the following scripts to compute various metrics:
     - `1-sensitivity.py` for recall values.
     - `2-precision.py` for precision values.
     - `3-specificity.py` for specificity values.
     - `4-f1_score.py` for F1 scores.

3. **Integrate Metrics**
   - Combine these metrics to gain insights into model performance for each class and overall.

---

## Requirements
- Python 3.x
- NumPy

---

## Author
- Davis Joseph

