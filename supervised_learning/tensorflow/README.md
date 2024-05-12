Tensorflow
 Master
 By: Alexa Orrico, Software Engineer at Holberton School
 Weight: 3
 Migrated to checker v2: 
 Your score will be updated as you progress.


Resources
Read or watch:

Low Level Intro (Excluding Datasets and Feature columns)
Graphs
Tensors
Variables
Placeholders
Save and Restore (Up to Save and restore models, excluded)
TensorFlow, why there are 3 files after saving the model?
Exporting and Importing a MetaGraph
TensorFlow - import meta graph and use variables from it
References:

tf.Graph
tf.Session
tf.Session.run
tf.Tensor
tf.Variable
tf.constant
tf.placeholder
tf.Operation
tf.keras.layers
tf.keras.layers.Dense
tf.keras.initializers.VarianceScaling
tf.nn
tf.nn.relu
tf.nn.sigmoid
tf.nn.tanh
tf.losses
tf.losses.softmax_cross_entropy
tf.train
tf.train.import_meta_graph
tf.train.GradientDescentOptimizer
tf.train.GradientDescentOptimizer.minimize
tf.train.Saver
tf.train.Saver.save
tf.train.Saver.restore
tf.add_to_collection
tf.get_collection
tf.global_variables_initializer
tf.argmax
tf.math.equal
tf.set_random_seed
tf.keras.backend.name_scope
Learning Objectives
At the end of this project, you are expected to be able to explain to anyone, without the help of Google:

General
What is tensorflow?
What is a session? graph?
What are tensors?
What are variables? constants? placeholders? How do you use them?
What are operations? How do you use them?
What are namespaces? How do you use them?
How to train a neural network in tensorflow
What is a checkpoint?
How to save/load a model with tensorflow
What is the graph collection?
How to add and get variables from the collection
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
Unless otherwise noted, you are not allowed to import any module except import tensorflow.compat.v1 as tf
All your files must be executable
The length of your files will be tested using wc
More Info
Installing Tensorflow 2.15
$ pip install --user tensorflow==2.15
Optimize Tensorflow (Optional)
to make use of your GPU, follow the steps in the tensorflow official website.
This will make training MUCH faster!

Note
During this project, your main task is to delve into Tensorflow v1. Be sure to anticipate the upcoming project on Tensorflow 2 & Keras, which will enhance your understanding of Tensorflow v2.

After completing this project, you can further explore the differences betwen Tensorflow v1 and Tensorflow v2 by consulting the following resources:

TensorFlow 1.x vs TensorFlow 2 - Behaviors and APIs
Tensorflow 1.xvs. Tensorflow 2.x: What’s the Difference?
