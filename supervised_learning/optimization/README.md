Resources
Read or watch:

Hyperparameter (machine learning)
Feature scaling
Why, How and When to Scale your Features
Normalizing your data
Moving average
An overview of gradient descent optimization algorithms
A Gentle Introduction to Mini-Batch Gradient Descent and How to Configure Batch Size
Stochastic Gradient Descent with momentum
Understanding RMSprop
Adam
Learning Rate Schedules
deeplearning.ai videos (Note: I suggest watching these video at 1.5x - 2x speed):
Normalizing Inputs
Mini Batch Gradient Descent
Understanding Mini-Batch Gradient Descent
Exponentially Weighted Averages
Understanding Exponentially Weighted Averages
Bias Correction of Exponentially Weighted Averages
Gradient Descent With Momentum
RMSProp
Adam Optimization Algorithm
Learning Rate Decay
Normalizing Activations in a Network
Fitting Batch Norm Into Neural Networks
Why Does Batch Norm Work?
Batch Norm At Test Time
The Problem of Local Optima
References:

numpy.random.permutation
tf.nn.moments
tf.keras.optimizers.SGD
tf.keras.optimizers.RMSprop
tf.keras.optimizers.Adam
tf.nn.batch_normalization
tf.keras.optimizers.schedules.InverseTimeDecay
Learning Objectives
At the end of this project, you are expected to be able to explain to anyone, without the help of Google:

General
What is a hyperparameter?
How and why do you normalize your input data?
What is a saddle point?
What is stochastic gradient descent?
What is mini-batch gradient descent?
What is a moving average? How do you implement it?
What is gradient descent with momentum? How do you implement it?
What is RMSProp? How do you implement it?
What is Adam optimization? How do you implement it?
What is learning rate decay? How do you implement it?
What is batch normalization? How do you implement it?
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
Unless otherwise noted, you are not allowed to import any module except import numpy as np and/or import tensorflow as tf
You should not import any module unless it is being used
All your files must be executable
The length of your files will be tested using wc
More Info
Please use the following model to accompany the tensorflow main files. You do not need to push this file to GitHub.

model.h5
