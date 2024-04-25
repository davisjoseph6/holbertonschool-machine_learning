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
        return f"leaf [value={self.value}]"


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
