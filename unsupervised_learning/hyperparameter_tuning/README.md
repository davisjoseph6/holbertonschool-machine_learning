# Hyperparameter Tuning for Machine Learning Models

This project is focused on hyperparameter tuning using Gaussian Processes (GP) and Bayesian Optimization to optimize the performance of machine learning models. It includes various scripts for performing Bayesian optimization and saving the best-performing models based on a set of hyperparameters.

---

## Directory Overview

### Files and Functions

1. **`0-gp.py`**
   - Contains the initial implementation of Gaussian Processes for hyperparameter optimization.
   
2. **`1-gp.py`**
   - Implements a more advanced version of Gaussian Processes, including optimization for a wider range of hyperparameters.

3. **`2-gp.py`**
   - Further improvements to Gaussian Processes with added functionalities for better optimization.

4. **`3-bayes_opt.py`**
   - Implements the core of Bayesian Optimization for model hyperparameter tuning.

5. **`4-bayes_opt.py`**
   - Continues the optimization process with additional methods or modifications for better results.

6. **`5-bayes_opt.py`**
   - Implements further optimization and tuning of hyperparameters using Bayesian methods.

7. **`bayes_opt.txt`**
   - Contains the configuration or parameters used during the Bayesian optimization process.

8. **`checkpoint_lr_<parameters>.h5`**
   - Various model checkpoints with different hyperparameters. Each checkpoint represents a saved model that corresponds to specific learning rate, number of units, dropout rate, L2 regularization, and batch size settings.
   - Example format: `checkpoint_lr_<learning_rate>_units_<num_units>_dropout_<dropout_rate>_l2_<l2_reg>_batch_<batch_size>.h5`

### Main Scripts

- **`0-main.py`** to **`6-main.py`**
   - These scripts provide the main logic to perform Bayesian optimization and evaluate the models using the different hyperparameter configurations provided in the `.py` files.

---

## Key Concepts

### Gaussian Processes (GP)
- Gaussian Processes are used for regression and optimization tasks. In this project, they are applied for hyperparameter optimization, where the goal is to find the best combination of hyperparameters that maximize the model's performance.

### Bayesian Optimization
- A probabilistic model-based optimization technique used to find the optimal hyperparameters for machine learning models. This method uses Gaussian Processes to model the objective function and make decisions about which hyperparameters to evaluate next.

---

## How to Use

1. **Run Gaussian Process Optimizer**
   - Execute the `gp.py` scripts (`0-gp.py`, `1-gp.py`, etc.) to train the model with different hyperparameters using Gaussian Processes.
   
2. **Perform Bayesian Optimization**
   - Use the `bayes_opt.py` scripts to optimize the hyperparameters by running Bayesian optimization.
   - The script iterates over different hyperparameter combinations and stores the best-performing models in the checkpoint files.

3. **Load and Use Checkpoints**
   - The models saved as `.h5` checkpoint files can be loaded using the `Keras` library. You can load the models based on the specific hyperparameters in the file name, such as:
     ```python
     from tensorflow.keras.models import load_model
     model = load_model('checkpoint_lr_0.005431706795656896_units_32_dropout_0.2056839462875551_l2_0.009090276159322614_batch_64.h5')
     ```

---

## Files

### Example Checkpoint Files
- **`checkpoint_lr_0.005431706795656896_units_32_dropout_0.2056839462875551_l2_0.009090276159322614_batch_64.h5`**
   - This checkpoint file contains a model trained with:
     - Learning rate: 0.0054
     - Units: 32
     - Dropout rate: 0.2057
     - L2 regularization: 0.0091
     - Batch size: 64

### **`bayes_opt.txt`**
- This file contains the hyperparameter ranges used during the Bayesian optimization process.

---

## Requirements

- Python 3.x
- Keras and TensorFlow for model training and checkpoint handling.
- NumPy and SciPy for optimization processes.
- Scikit-learn for additional machine learning utilities.

---

## Applications
- Hyperparameter tuning for deep learning models.
- Optimization of machine learning models to improve accuracy.
- Model selection for various machine learning tasks, including regression, classification, and time series forecasting.

---

## Author
- Davis Joseph ([LinkedIn](https://www.linkedin.com/in/davisjoseph767/))


