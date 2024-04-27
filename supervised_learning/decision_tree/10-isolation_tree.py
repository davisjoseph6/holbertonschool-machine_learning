#!/usr/bin/env python3

"""
This is the 10-isolation_tree module, utilizing a custom tree
structure for outlier detection.
"""

import numpy as np
Node = __import__('8-build_decision_tree').Node
Leaf = __import__('8-build_decision_tree').Leaf


class Isolation_Random_Tree():
    """
    A class representing an Isolation tree, specifically designed for
    outlier detection.
    """
    def __init__(self, max_depth=10, seed=0, root=None):
        # Random number generator for reproducibility
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)  # Starting root node
        self.explanatory = None
        self.max_depth = max_depth
        self.predict = None
        self.min_pop = 1

    def __str__(self):
        """
        String representation for the tree object.
        """
        return f"{self.root.__str__()}\n"

    def depth(self):
        """
        Returns the maximum depth of the isolation tree.
        """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """
        Returns the number of nodes in the isolation tree.
        If only_leaves is True, only counts leaf nodes.
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def update_bounds(self):
        """
        Calls update_bounds_below() on the root.
        """
        self.root.update_bounds_below()

    def get_leaves(self):
        """
        Retrieves the list of leaves in the tree.
        """
        return self.root.get_leaves_below()

    def update_predict(self):
        """
        Updates the prediction function of the isolation tree.
        """
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()

        self.predict = lambda A: np.array([self.root.pred(x) for x in A])

    def np_extrema(self, arr):
        """
        Returns the minimum and maximum values of the array.
        """
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """
        Determines the split criterion randomly for the node.
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

    def get_leaf_child(self, node, sub_population):
        """
        Creates a leaf child node with specified sub_population and
        increased depth.
        """
        leaf_child = Leaf(node.depth + 1)
        leaf_child.subpopulation = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """
        Creates a non-leaf child node for further splits.
        """
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def fit_node(self, node):
        """
        Recursively fits nodes of the tree by determining whether to create
        a leaf or continue splitting.
        """
        node.feature, node.threshold = self.random_split_criterion(node)

        max_criterion = np.greater(
                self.explanatory[:, node.feature], node.threshold)

        left_population = np.logical_and(node.sub_population, max_criterion)

        right_population = np.logical_and(
                node.sub_population, np.logical_not(max_criterion))

        is_left_leaf = (node.depth == self.max_depth - 1 or
                        np.sum(left_population) <= self.min_pop)

        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        is_right_leaf = (node.depth == self.max_depth - 1 or
                         np.sum(right_population) <= self.min_pop)

        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def fit(self, explanatory, verbose=0):
        """
        Fits the isolation tree to the dataset
        """
        self.explanatory = explanatory
        self.root.sub_population = np.ones(explanatory.shape[0], dtype=bool)
        self.fit_node(self.root)
        self.update_predict()

        if verbose == 1:
            print(f"Training finished.\n"
                  f" - Depth: {self.depth()}\n"
                  f" - Number of nodes: {self.count_nodes()}\n"
                  f" - Number of leaves: {self.count_nodes(only_leaves=True)}")
