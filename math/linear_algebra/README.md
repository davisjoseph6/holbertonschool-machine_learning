# Linear Algebra Utilities

## Overview

This project provides a collection of Python scripts and functions to perform various operations in linear algebra, ranging from slicing arrays to matrix multiplication. These utilities are implemented using both basic Python constructs and the `numpy` library.

## Directory Structure

The project is organized as follows:
```
. ├── 0-slice_me_up.py # Slicing a Python list in different ways 
  ├── 1-trim_me_down.py # Extracting specific parts of a matrix 
  ├── 2-size_me_please.py # Determining the shape of a Python list-based matrix 
  ├── 3-flip_me_over.py # Transposing a 2D matrix 
  ├── 4-line_up.py # Adding two arrays element-wise 
  ├── 5-across_the_planes.py # Adding two 2D matrices element-wise 
  ├── 6-howdy_partner.py # Concatenating two arrays 
  ├── 7-gettin_cozy.py # Concatenating two matrices along a specific axis 
  ├── 8-ridin_bareback.py # Performing matrix multiplication 
  ├── 9-let_the_butcher_slice_it.py # Slicing a NumPy array 
  ├── 10-ill_use_my_scale.py # Determining the shape of a NumPy array 
  ├── 11-the_western_exchange.py # Transposing a NumPy array 
  ├── 12-bracin_the_elements.py # Performing element-wise operations on NumPy arrays 
  ├── 13-cats_got_your_tongue.py # Concatenating NumPy arrays along a specific axis 
  ├── 14-saddle_up.py # Performing matrix multiplication with NumPy 
  ├── 100-slice_like_a_ninja.py # Slicing a NumPy array along multiple axes 
  ├── 101-the_whole_barn.py # Adding n-dimensional matrices with the same shape 
  ├── 102-squashed_like_sardines.py # Concatenating n-dimensional matrices 
  ├── README.md # Project documentation 
  ├── init.py # Package initialization └── tests/ # Unit tests for the functions
```

## Features

### Basic Python Implementations

- **Slicing and Manipulating Lists and Matrices**:
  - `0-slice_me_up.py`: Demonstrates slicing of lists.
  - `1-trim_me_down.py`: Extracts specific parts of a 2D matrix.
  - `2-size_me_please.py`: Determines the shape of a nested list structure.
  - `3-flip_me_over.py`: Transposes a 2D matrix.

- **Element-wise Operations**:
  - `4-line_up.py`: Adds two lists element-wise.
  - `5-across_the_planes.py`: Adds two 2D matrices element-wise.
  - `6-howdy_partner.py`: Concatenates two lists.
  - `7-gettin_cozy.py`: Concatenates two matrices along a specified axis.

- **Matrix Operations**:
  - `8-ridin_bareback.py`: Multiplies two matrices using basic Python.

### NumPy-based Implementations

- **Array Manipulation**:
  - `9-let_the_butcher_slice_it.py`: Slices NumPy arrays.
  - `10-ill_use_my_scale.py`: Determines the shape of a NumPy array.
  - `11-the_western_exchange.py`: Transposes a NumPy array.
  - `12-bracin_the_elements.py`: Performs element-wise arithmetic on NumPy arrays.
  - `13-cats_got_your_tongue.py`: Concatenates NumPy arrays along a specific axis.
  - `14-saddle_up.py`: Multiplies two NumPy arrays using the `@` operator.

- **Advanced Slicing and Concatenation**:
  - `100-slice_like_a_ninja.py`: Slices NumPy arrays along multiple axes.
  - `101-the_whole_barn.py`: Adds n-dimensional matrices with the same shape.
  - `102-squashed_like_sardines.py`: Concatenates n-dimensional matrices along a specific axis.

## Getting Started

### Prerequisites

- Python 3.4 or later
- NumPy

### Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/username/holbertonschool-machine_learning.git
   cd holbertonschool-machine_learning/math/linear_algebra
   ```

2. Run any script directly:

```bash
./0-slice_me_up.py
```

Import functions in your Python scripts:

```python
from 2-size_me_please import matrix_shape

matrix = [[1, 2, 3], [4, 5, 6]]
print(matrix_shape(matrix))  # Output: [2, 3]
```

## Testing
Unit tests are available in the `tests/` directory. Run them using:

```bash
python3 -m unittest discover tests
```

## Author
Davis Joseph

## License
This project is licensed under the Holberton School guidelines.
