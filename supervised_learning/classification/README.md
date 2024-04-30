Classification Using Neural Networks
 Amateur
 By: Alexa Orrico, Software Engineer at Holberton School
 Weight: 3
 Migrated to checker v2: 
 Your score will be updated as you progress.
 Manual QA review must be done (request it when you are done with the project)


Background Context
At the end of this project, you should be able to build your own binary image classifier from scratch using numpy.

Good luck and have fun!

Resources
Read or watch:

Supervised vs. Unsupervised Machine Learning
How would you explain neural networks to someone who knows very little about AI or neurology?
Using Neural Nets to Recognize Handwritten Digits (until “A simple network to classify handwritten digits” (excluded))
Forward propagation
Understanding Activation Functions in Neural Networks
Loss function
Gradient descent
Calculus on Computational Graphs: Backpropagation
Backpropagation calculus
What is a Neural Network?
Supervised Learning with a Neural Network
Binary Classification
Logistic Regression
Logistic Regression Cost Function
Gradient Descent
Computation Graph
Logistic Regression Gradient Descent
Vectorization
Vectorizing Logistic Regression
Vectorizing Logistic Regression’s Gradient Computation
A Note on Python/Numpy Vectors
Neural Network Representations
Computing Neural Network Output
Vectorizing Across Multiple Examples
Gradient Descent For Neural Networks
Random Initialization
Deep L-Layer Neural Network
Train/Dev/Test Sets
Random Initialization For Neural Networks : A Thing Of The Past
Initialization of deep networks
Multiclass classification
Derivation: Derivatives for Common Neural Network Activation Functions
What is One Hot Encoding? Why And When do you have to use it?
Softmax function
What is the intuition behind SoftMax function?
Cross entropy
Loss Functions: Cross-Entropy
Softmax Regression (Note: I suggest watching this video at 1.5x - 2x speed)
Training Softmax Classifier (Note: I suggest watching this video at 1.5x - 2x speed)
numpy.zeros
numpy.random.randn
numpy.exp
numpy.log
numpy.sqrt
numpy.where
numpy.max
numpy.sum
numpy.argmax
What is Pickle in python?
pickle
pickle.dump
pickle.load
Optional:

Predictive analytics
Maximum Likelihood Estimation
Learning Objectives
At the end of this project, you are expected to be able to explain to anyone, without the help of Google:

General
What is a model?
What is supervised learning?
What is a prediction?
What is a node?
What is a weight?
What is a bias?
What are activation functions?
Sigmoid?
Tanh?
Relu?
Softmax?
What is a layer?
What is a hidden layer?
What is Logistic Regression?
What is a loss function?
What is a cost function?
What is forward propagation?
What is Gradient Descent?
What is back propagation?
What is a Computation Graph?
How to initialize weights/biases
The importance of vectorization
How to split up your data
What is multiclass classification?
What is a one-hot vector?
How to encode/decode one-hot vectors
What is the softmax function and when do you use it?
What is cross-entropy loss?
What is pickling in Python?
Requirements
General
Allowed editors: vi, vim, emacs
All your files will be interpreted/compiled on Ubuntu 20.04 LTS using python3 (version 3.9)
Your files will be executed with numpy (version 1.25.2)
All your files should end with a new line
The first line of all your files should be exactly #!/usr/bin/env python3
A README.md file, at the root of the folder of the project, is mandatory
Your code should use the pycodestyle style (version 2.11.1)
All your modules should have documentation (python3 -c 'print(__import__("my_module").__doc__)')
All your classes should have documentation (python3 -c 'print(__import__("my_module").MyClass.__doc__)')
All your functions (inside and outside a class) should have documentation (python3 -c 'print(__import__("my_module").my_function.__doc__)' and python3 -c 'print(__import__("my_module").MyClass.my_function.__doc__)')
Unless otherwise noted, you are not allowed to import any module except import numpy as np
Unless otherwise noted, you are not allowed to use any loops (for, while, etc.)
All your files must be executable
The length of your files will be tested using wc
More Info
Matrix Multiplications
For all matrix multiplications in the following tasks, please use numpy.matmul

Testing your code
In order to test your code, you’ll need DATA! Please download these datasets (Binary_Train.npz, Binary_Dev.npz, MNIST.npz) to go along with all of the following main files. You do not need to upload these files to GitHub. Your code will not necessarily be tested with these datasets. All of the following code assumes that you have stored all of your datasets in a separate data directory.

alexa@ubuntu-xenial:$ cat show_data.py
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

lib_train = np.load('../data/Binary_Train.npz')
X_3D, Y = lib_train['X'], lib_train['Y']

fig = plt.figure(figsize=(10, 10))
for i in range(100):
    fig.add_subplot(10, 10, i + 1)
    plt.imshow(X_3D[i])
    plt.title(Y[0, i])
    plt.axis('off')
plt.tight_layout()
plt.show()
alexa@ubuntu-xenial:$ ./show_data.py


alexa@ubuntu-xenial:$ cat show_multi_data.py
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

lib = np.load('../data/MNIST.npz')
print(lib.files)
X_train_3D = lib['X_train']
Y_train = lib['Y_train']

fig = plt.figure(figsize=(10, 10))
for i in range(100):
    fig.add_subplot(10, 10, i + 1)
    plt.imshow(X_train_3D[i])
    plt.title(str(Y_train[i]))
    plt.axis('off')
plt.tight_layout()
plt.show()
alexa@ubuntu-xenial:$ ./show_multi_data.py
['Y_test', 'X_test', 'X_train', 'Y_train', 'X_valid', 'Y_valid']

