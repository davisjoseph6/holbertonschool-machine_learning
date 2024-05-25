Resources
Read or watch:

Regularization (mathematics)
An Overview of Regularization Techniques in Deep Learning (up to A case study on MNIST data with keras excluded)
L2 Regularization and Back-Propagation
Intuitions on L1 and L2 Regularisation
Analysis of Dropout
Early stopping
How to use early stopping properly for training deep neural network?
Data Augmentation | How to use Deep Learning when you have Limited Data 
deeplearning.ai videos (Note: I suggest watching these video at 1.5x - 2x speed):
Regularization
Why Regularization Reduces Overfitting
Dropout Regularization
Understanding Dropout
Other Regularization Methods
References:

numpy.linalg.norm
numpy.random.binomial
tf.keras.regularizers.L2
tf.keras.layers.Dense
Regularization loss
tf.keras.layers.Dropout
Dropout: A Simple Way to Prevent Neural Networks from Overfitting
Early Stopping - but when?
L2 Regularization versus Batch and Weight Normalization
Learning Objectives
At the end of this project, you are expected to be able to explain to anyone, without the help of Google:

General
What is regularization? What is its purpose?
What is are L1 and L2 regularization? What is the difference between the two methods?
What is dropout?
What is early stopping?
What is data augmentation?
How do you implement the above regularization methods in Numpy? Tensorflow?
What are the pros and cons of the above regularization methods?
Requirements
General
Allowed editors: vi, vim, emacs
All your files will be interpreted/compiled on Ubuntu 20.04 LTS using python3 (version 3.9)
Your files will be executed with numpy (version 1.25.2) and tensorflow (version 2.15)
All your files should end with a new line
The first line of all your files should be exactly #!/usr/bin/env python3
A README.md file, at the root of the folder of the project, is mandatory
Your code should use the pycodestyle style (version 2.11.1)
All your modules should have documentation (python3 -c 'print(__import__("my_module").__doc__)')
All your classes should have documentation (python3 -c 'print(__import__("my_module").MyClass.__doc__)')
All your functions (inside and outside a class) should have documentation (python3 -c 'print(__import__("my_module").my_function.__doc__)' and python3 -c 'print(__import__("my_module").MyClass.my_function.__doc__)')
Unless otherwise noted, you are not allowed to import any module except import numpy as np and import tensorflow as tf
You should not import any module unless it is being used
All your files must be executable
The length of your files will be tested using wc
When initializing layer weights, use tf.keras.initializers.VarianceScaling(scale=2.0, mode=("fan_avg")).
