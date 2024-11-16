# Advanced Linear Algebra

This project focuses on implementing various advanced linear algebra operations, including determinants, minors, cofactors, adjugates, inverses, and definiteness of matrices. These mathematical tools are fundamental in many areas of computational science and engineering, including machine learning, optimization, and computer graphics.

---

## Directory Overview

### Matrix Operations
1. **`0-determinant.py`**  
   - Calculates the determinant of a matrix.
   - Includes validation for square matrices and handles edge cases such as empty or 1x1 matrices.

2. **`1-minor.py`**  
   - Computes the minor matrix, where each element is the determinant of a submatrix formed by removing one row and one column.

3. **`2-cofactor.py`**  
   - Computes the cofactor matrix by applying a sign factor to the minor matrix.

4. **`3-adjugate.py`**  
   - Calculates the adjugate (adjoint) of a matrix, which is the transpose of the cofactor matrix.

5. **`4-inverse.py`**  
   - Computes the inverse of a matrix if it exists. Utilizes the adjugate and determinant of the matrix for calculation.

6. **`5-definiteness.py`**  
   - Determines the definiteness of a square, symmetric matrix based on its eigenvalues.  
   - Outputs:
     - **Positive definite**
     - **Positive semi-definite**
     - **Negative definite**
     - **Negative semi-definite**
     - **Indefinite**

---

## Features

### Validation
- Ensures all inputs are valid matrices, handling exceptions for non-square, non-numeric, or empty matrices.

### Recursive Determinant Calculation
- The determinant function uses a recursive approach with cofactor expansion, allowing it to handle matrices of arbitrary size.

### Eigenvalue Analysis
- Eigenvalues are computed for definiteness classification, leveraging NumPy's linear algebra capabilities.

---

## Applications
- **Machine Learning**: Matrix inversions and determinants are used in algorithms like linear regression and Gaussian processes.
- **Optimization**: Definiteness checks ensure a matrix is suitable for optimization problems.
- **Computer Graphics**: Transformation matrices require determinant and inverse operations.

---

## How to Use

### Example: Determinant
```python
from 0-determinant import determinant

matrix = [[1, 2], [3, 4]]
det = determinant(matrix)
print("Determinant:", det)
```

### Example: Minor Matrix
```python
from 1-minor import minor

matrix = [[1, 2], [3, 4]]
minors = minor(matrix)
print("Minor Matrix:", minors)
```

### Example: Inverse
```python
from 4-inverse import inverse

matrix = [[4, 7], [2, 6]]
inv = inverse(matrix)
print("Inverse Matrix:", inv)
```

### Example: Definiteness
```python
from 5-definiteness import definiteness
import numpy as np

matrix = np.array([[2, -1], [-1, 2]])
def_type = definiteness(matrix)
print("Definiteness:", def_type)
```

## Requirements
- Python 3.x
- NumPy

### References
- Linear Algebra and Its Applications, Gilbert Strang
- Numerical Recipes: The Art of Scientific Computing

