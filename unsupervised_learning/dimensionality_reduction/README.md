Dimensionality Reduction
 Master
 By: Alexa Orrico, Software Engineer at Holberton School
 Weight: 2
 Migrated to checker v2: 
 Your score will be updated as you progress.
 Manual QA review must be done (request it when you are done with the project)


Resources
Read or watch:

Dimensionality Reduction For Dummies — Part 1: Intuition
Singular Value Decomposition
Understanding SVD (Singular Value Decomposition)
Intuitively, what is the difference between Eigendecomposition and Singular Value Decomposition?
Dimensionality Reduction: Principal Components Analysis, Part 1
Dimensionality Reduction: Principal Components Analysis, Part 2
StatQuest: t-SNE, Clearly Explained
t-SNE tutorial Part1
t-SNE tutorial Part2
How to Use t-SNE Effectively
Definitions to skim:

Dimensionality Reduction
Principal component analysis
Eigendecomposition of a matrix
Singular value decomposition
Manifold check this out if you have never heard this term before
Kullback–Leibler divergence
T-distributed stochastic neighbor embedding
As references:

numpy.cumsum
Visualizing Data using t-SNE (paper)
Visualizing Data Using t-SNE (video)
Advanced:

Kernel principal component analysis
Nonlinear Dimensionality Reduction: KPCA
Learning Objectives
What is eigendecomposition?
What is singular value decomposition?
What is the difference between eig and svd?
What is dimensionality reduction and what are its purposes?
What is principal components analysis (PCA)?
What is t-distributed stochastic neighbor embedding (t-SNE)?
What is a manifold?
What is the difference between linear and non-linear dimensionality reduction?
Which techniques are linear/non-linear?
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
All your files must be executable
Your code should use the minimum number of operations to avoid floating point errors
Data
Please test your main files with the following data:

mnist2500_X.txt
mnist2500_labels.txt
Watch Out!
Just like lists, np.ndarrays are mutable objects:

>>> vector = np.ones((100, 1))
>>> m1 = vector[55]
>>> m2 = vector[55, 0]
>>> vector[55] = 2
>>> m1
array([2.])
>>> m2
1.0
Performance between SVD and EIG
Here a graph of execution time (Y-axis) for the number of iteration (X-axis) - red line is EIG and blue line is SVG


