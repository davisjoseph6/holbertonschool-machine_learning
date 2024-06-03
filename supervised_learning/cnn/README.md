Convolutional Neural Networks
 Master
 By: Alexa Orrico, Software Engineer at Holberton School
 Weight: 6
 Migrated to checker v2: 
 Your score will be updated as you progress.
 Manual QA review must be done (request it when you are done with the project)


Resources
Read or watch:

Convolutional neural network
Convolutional Neural Networks (CNNs) explained
The best explanation of Convolutional Neural Networks on the Internet! (It’s pretty good but I wouldn’t call it the best…)
Machine Learning is Fun! Part 3: Deep Learning and Convolutional Neural Networks
Convolutional Neural Networks: The Biologically-Inspired Model
Back Propagation in Convolutional Neural Networks — Intuition and Code
Backpropagation in a convolutional layer
Convolutional Neural Network – Backward Propagation of the Pooling Layers
Pooling Layer
deeplearning.ai videos (Note: I suggest watching these videos at 1.5x - 2x speed):
Why Convolutions
One Layer of a Convolutional Net
Simple Convolutional Network Example
CNN Example
Gradient-Based Learning Applied to Document Recognition (LeNet-5)
References:

tf.layers.Conv2D
tf.keras.layers.Conv2D
tf.layers.AveragePooling2D
tf.keras.layers.AveragePooling2D
tf.layers.MaxPooling2D
tf.keras.layers.MaxPooling2D
tf.layers.Flatten
tf.keras.layers.Flatten
Reproducibility in Keras Models
Learning Objectives
At the end of this project, you are expected to be able to explain to anyone, without the help of Google:

General
What is a convolutional layer?
What is a pooling layer?
Forward propagation over convolutional and pooling layers
Back propagation over convolutional and pooling layers
How to build a CNN using Tensorflow and Keras
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
Unless otherwise noted, you are not allowed to import any module
All your files must be executable.
The length of your files will be tested using wc
