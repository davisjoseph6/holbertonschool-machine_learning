Decision Tree & Random Forest
 Novice
 By: HBTN
 Weight: 1
 Migrated to checker v2: 
 Your score will be updated once you launch the project review.
Concepts
For this project, we expect you to look at these concepts:

What is a decision tree?
Decision_Tree.pred vs Decision_Tree.predict





Resources

Read or watch:
(1) Rokach and Maimon (2002) : Top-down induction of decision trees classifiers : a survey
(2) Ho et al. (1995) : Random Decision Forests
(3) Fei et al. (2008) : Isolation forests
(4) Gini and Entropy clearly explained : Handling Continuous features in Decision Trees
(5) Abspoel and al. (2021) : Secure training of decision trees with continuous attributes
(6) Threshold Split Selection Algorithm for Continuous Features in Decision Tree
(7) Splitting Continuous Attribute using Gini Index in Decision Tree
(8) How to handle Continuous Valued Attributes in Decision Tree
(9) Decision Tree problem based on the Continuous-valued attribute
(10) How to Implement Decision Trees in Python using Scikit-Learn(sklearn)
(11)Matching and Prediction on the Principle of Biological Classification by William A. Belson

Notes
This project aims to implement decision trees from scratch. It is important for engineers to understand how the tools we use are built for two reasons. First, it gives us confidence in our skills. Second, it helps us when we need to build our own tools to solve unsolved problems.
The first three references point to historical papers where the concepts were first studied.
References 4 to 9 can help if you feel you need some more explanation about the way we split nodes.
William A. Belson is usually credited for the invention of decision trees (read reference 11).
Despite our efforts to make it efficient, we cannot compete with Sklearn’s implementations (since they are done in C). In real life, it is thus recommended to use Sklearn’s tools.
In this regard, it is warmly recommended to watch the video referenced as (10) above. It shows how to use Sklearn’s decision trees and insists on the methodology.

Tasks

We will progressively add methods in the following 3 classes :

class Node:
    def __init__(self, feature=None, threshold=None, left_child=None, right_child=None, is_root=False, depth=0):
        self.feature                  = feature
        self.threshold                = threshold
        self.left_child               = left_child
        self.right_child              = right_child
        self.is_leaf                  = False
        self.is_root                  = is_root
        self.sub_population           = None    
        self.depth                    = depth
                
class Leaf(Node):
    def __init__(self, value, depth=None) :
        super().__init__()
        self.value   = value
        self.is_leaf = True
        self.depth   = depth

class Decision_Tree() :
    def __init__(self, max_depth=10, min_pop=1, seed=0,split_criterion="random", root=None) :
        self.rng               = np.random.default_rng(seed)
        if root :
            self.root          = root
        else :
            self.root          = Node(is_root=True)
        self.explanatory       = None
        self.target            = None
        self.max_depth         = max_depth
        self.min_pop           = min_pop
        self.split_criterion   = split_criterion
        self.predict           = None
        


Once built, decision trees are binary trees : a node either is a leaf or has two children. It never happens that a node for which is_leaf is False has its left_child or right_child left unspecified.
The first three tasks are a warm-up designed to review the basics of class inheritance and recursion (nevertheless, the functions coded in these tasks will be reused in the rest of the project).
Our first objective will be to write a Decision_Tree.predict method that takes the explanatory features of a set of individuals and returns the predicted target value for these individuals.
Then we will write a method Decision_Tree.fit that takes the explanatory features and the targets of a set of individuals, and grows the tree from the root to the leaves to make it in an efficient prediction tool.
Once these tasks will be accomplished, we will introduce a new class Random_Forest that will also be a powerful prediction tool.
Finally, we will write a variation on Random_Forest, called Isolation_Random_forest, that will be a tool to detect outliers.

Requirements

You should carefully read all the concept pages attached above.
All your files will be interpreted/compiled on Ubuntu 20.04 LTS using python3 (version 3.9)
Your files will be executed with numpy (version 1.25.2)
All your files should end with a new line
Your code should use the pycodestyle style (version 2.11.1)
The first line of all your files should be exactly #!/usr/bin/env python3
A README.md file, at the root of the folder of the project, is mandatory
All your modules should have documentation (python3 -c 'print(__import__("my_module").__doc__)')
All your classes should have documentation (python3 -c 'print(__import__("my_module").MyClass.__doc__)')
All your functions (inside and outside a class) should have documentation (python3 -c 'print(__import__("my_module").my_function.__doc__)' and python3 -c 'print(__import__("my_module").MyClass.my_function.__doc__)')
All your files must be executable
