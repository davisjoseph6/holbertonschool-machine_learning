#!/usr/bin/env python3

"""
Decision Tree Components
Includes classes for nodes (both decision and leaf nodes) and the
decision tree itself.
"""
import numpy as np


class Node:
    """
    Represents a decision node in a decision tree, which can split data based
    on features and thresholds.
    """
    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        """
        Initializes the node with optional feature splits, threshold values,
        children, root status, and depth.
        """
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth
        self.lower = None  # Initialize to None, to be updated
        self.upper = None  # Initialize to None, to be updated

    def max_depth_below(self):
        """
        Returns the maximum depth of the tree beneath this node.
        """
        max_depth = self.depth

        # If the node has a left child, calculate the maximum depth below
        # the left child
        if self.left_child is not None:
            max_depth = max(max_depth, self.left_child.max_depth_below())

        # If the node has a right child, calculate the maximum depth below
        # the right child
        if self.right_child is not None:
            max_depth = max(max_depth, self.right_child.max_depth_below())

        return max_depth

    def count_nodes_below(self, only_leaves=False):
        """
        Counts the nodes in the subtree rooted at this node.
        Optionally counts only leaf nodes.
        """
        if only_leaves:
            # If only leaves should be counted, skip counting for non-leaf
            # nodes.
            if self.is_leaf:
                return 1
            count = 0
        else:
            # Count this node if we are not only counting leaves
            count = 1

        # Recursively count the nodes int the left and right subtress
        if self.left_child is not None:
            count += self.left_child.count_nodes_below(only_leaves)
        if self.right_child is not None:
            count += self.right_child.count_nodes_below(only_leaves)

        return count

    def __str__(self):
        """
        Returns a string representation of the node and it's children
        """
        node_type = "root" if self.is_root else "node"
        details = (f"{node_type} [feature={self.feature}, "
                   f"threshold={self.threshold}]\n")
        if self.left_child:
            left_str = self.left_child.__str__().replace("\n", "\n    |  ")
            details += f"    +---> {left_str}"

        if self.right_child:
            right_str = self.right_child.__str__().replace("\n", "\n       ")
            details += f"\n    +---> {right_str}"

        return details

    def get_leaves_below(self):
        """
        Returns a list of all leaves below this node.
        """
        leaves = []
        if self.left_child:
            leaves.extend(self.left_child.get_leaves_below())
        if self.right_child:
            leaves.extend(self.right_child.get_leaves_below())
        return leaves

    def update_bounds_below(self):
        """
        Recursively compute, for each node, two dictionaries stored as
        attributes Node.lower and Node.upper. These dictionaries contain
        the bounds for each feature.
        """
        if self.is_root:
            self.lower = {0: -np.inf}
            self.upper = {0: np.inf}

        if self.left_child:
            # Copy bounds from parent and update
            self.left_child.lower = self.lower.copy()
            self.left_child.upper = self.upper.copy()

            if self.feature in self.left_child.lower:
                # Update left child's lower bound for the feature
                self.left_child.lower[self.feature] = max(
                        self.threshold, self.left_child.lower[self.feature]
                        )
            else:
                self.left_child.lower[self.feature] = self.threshold

            # Recurse into the left child
            self.left_child.update_bounds_below()

        if self.right_child:
            # Copy bounds from parent and update
            self.right_child.lower = self.lower.copy()
            self.right_child.upper = self.upper.copy()

            if self.feature in self.right_child.upper:
                # Update right child's upper bound for the feature
                self.right_child.upper[self.feature] = min(
                        self.threshold, self.right_child.upper[self.feature]
                        )
            else:
                self.right_child.upper[self.feature] = self.threshold

            # Recurse into the right child
            self.right_child.update_bounds_below()

    def update_indicator(self):
        """
        Update the indicator function based on the lower and upper bounds.
        """
        def is_large_enough(x):
            """
            is large enough
            """
            comparisons = [x[:, key] > self.lower[key] for key in self.lower]
            return np.all(comparisons, axis=0)

        def is_small_enough(x):
            """
            is small enough
            """
            comparisons = [x[:, key] <= self.upper[key] for key in self.upper]
            return np.all(comparisons, axis=0)

        self.indicator = lambda x: (
                np.logical_and(is_large_enough(x), is_small_enough(x))
                )

    def pred(self, x):
        """
        Predict the class label for a single instance x
        based on the tree structure
        """
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        else:
            return self.right_child.pred(x)


class Leaf(Node):
    """
    Represents a leaf node in a decision tree, holding a constant value
    and depth.
    """
    def __init__(self, value, depth=None):
        """
        Initializes the leaf with a specific value and depth.
        """
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """
        Returns the depth of the leaf, as leaf nodes are the endpoints
        of a tree
        """
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """
        Returns 1 since leaves count as one node each.
        """
        return 1

    def __str__(self):
        """
        Returns a string representation of the leaf.
        """
        return f"-> leaf [value={self.value}] "

    def get_leaves_below(self):
        """
        Returns a list containing only this leaf.
        """
        return [self]

    def update_bounds_below(self):
        """
        Leaves do not need to update bounds as they represent endpoints
        """
        pass

    def pred(self, x):
        """
        def pred
        """
        return self.value


class Decision_Tree():
    """
    Implements a decision tree that can be used for various
    decision-making processes.
    """
    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """
        Initializes the decision tree with parameters for tree construction
        and random number generation.
        """
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """
        Returns the maximum depth of a tree
        """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """
        Counts the total nodes or only leaf nodes in the tree
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """
        Returns a string representation of the entire decision tree.
        """
        return self.root.__str__() + "\n"

    def get_leaves(self):
        """
        Retrieves all leaf nodes of the tree.
        """
        return self.root.get_leaves_below()

    def update_bounds(self):
        """
        Initiates the bounds update process from the root.
        """
        self.root.update_bounds_below()

    def update_predict(self):
        """
        Updates the prediction function for the decision tree.
        """
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()

        def predict(A):
            results = np.empty(A.shape[0], dtype=int)
            for leaf in leaves:
                indices = leaf.indicator(A)
                results[indices] = leaf.value
            return results

        self.predict = predict

    def pred(self, x):
        """
        def pred
        """
        return self.root.pred(x)

    def fit(self, explanatory, target, verbose=0):
        """
        Initializes some attributes of the tree and then calls a new method
        Decision_Tree.fit_node on the root.
        """
        if self.split_criterion == "random":
            self.split_criterion = self.random_split_criterion
        else:
            self.split_criterion = self.Gini_split_criterion
        self.explanatory = explanatory
        self.target = target
        self.root.sub_population = np.ones_like(self.target, dtype='bool')

        self.fit_node(self.root)

        self.update_predict()

        if verbose == 1:
            print(f"""  Training finished.
    - Depth                     : { self.depth()       }
    - Number of nodes           : { self.count_nodes() }
    - Number of leaves          : { self.count_nodes(only_leaves=True) }
    - Accuracy on training data : { self.accuracy(self.explanatory,
                                              self.target)}""")

    def np_extrema(self, arr):
        """
        Compute the minimum and maximum values of an array using NumPy.
        Returns the values as a tuple.
        """
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """
        Randomly selects a feature and threshold to split the node's
        subpopulation.

        Args:
            node (Node): The node to split.

        Returns:
            tuple: A tuple containing the selected feature and threshold.
        """
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            feature_min, feature_max = self.np_extrema(
                self.explanatory[:, feature][node.sub_population])
            diff = feature_max - feature_min
        x = self.rng.uniform()
        threshold = (1 - x) * feature_min + x * feature_max
        return feature, threshold

    def fit_node(self, node):
        """
        Fits a decision tree node by recursively splitting the data based on
        the best split criterion.
        """
        node.feature, node.threshold = self.split_criterion(node)

        max_criterion = np.greater(
            self.explanatory[:, node.feature],
            node.threshold)

        left_population = np.logical_and(
            node.sub_population,
            max_criterion)

        # "War does not determine who is right - only who is left."
        right_population = np.logical_and(
            node.sub_population,
            np.logical_not(max_criterion))

        # Is left node a leaf ?
        is_left_leaf = np.any(np.array(
            [node.depth == self.max_depth - 1,
             np.sum(left_population) <= self.min_pop,
             np.unique(self.target[left_population]).size == 1]))

        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        # Is right node a leaf ?
        is_right_leaf = np.any(np.array(
            [node.depth == self.max_depth - 1,
             np.sum(right_population) <= self.min_pop,
             np.unique(self.target[right_population]).size == 1]))

        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def get_leaf_child(self, node, sub_population):
        """
        Create a leaf child node with the most frequent target value in the
        given subpopulation and returns the new object.
        """
        value = np.argmax(np.bincount(self.target[sub_population]))
        leaf_child = Leaf(value)
        leaf_child.depth = node.depth + 1
        # NOTE this should be leaf_child.subpopulation_leaf
        leaf_child.subpopulation = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """
        Create a new child node for the given parent node.

        Args:
            node (Node): The parent node.
            sub_population (list): The sub-population associated with
                the child node.

        Returns:
            Node: The newly created child node.
        """
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def accuracy(self, test_explanatory, test_target):
        """
        Calculates the accuracy of the decision tree model on the given
        test data.

        Args:
        test_explanatory (numpy.ndarray): The explanatory variables of
            the test data.
        test_target (numpy.ndarray): The target variable of the test data.

        Returns:
        float: The accuracy of the decision tree model on the test data.
        """
        return np.sum(np.equal(
            self.predict(test_explanatory), test_target)) / test_target.size

