#!/usr/bin/env python3

"""
Decision Tree Components
Includes classes for nodes (both decision and leaf nodes) and the
decision tree itself
"""
import numpy as np

class Node:
    """
    Represents a node in a decision tree, used to split data based on features and thresholds.

    Attributes:
        feature (int or None): Index of the feature used for splitting.
        threshold (float or None): Threshold value for the split.
        left_child (Node or None): Left child node.
        right_child (Node or None): Right child node.
        is_leaf (bool): Indicates if the node is a leaf.
        is_root (bool): Indicates if the node is the root.
        depth (int): Depth of the node in the tree.
    """
    def __init__(self, feature=None, threshold=None, left_child=None, right_child=None, is_root=False, depth=0):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.depth = depth

    def max_depth_below(self):
        """
        Recursively calculates and returns the maximum depth of the subtree below this node.
        """
        if self.is_leaf:
            return self.depth
        max_depth = self.depth
        if self.left_child is not None:
            max_depth = max(max_depth, self.left_child.max_depth_below())
        if self.right_child is not None:
            max_depth = max(max_depth, self.right_child.max_depth_below())
        return max_depth

    def count_nodes_below(self, only_leaves=False):
        """
        Counts and returns the number of nodes in the subtree rooted at this node.
        If only_leaves is True, counts only the leaf nodes.
        """
        if only_leaves and self.is_leaf:
            return 1
        if self.is_leaf:
            return 1
        count = 1 if not only_leaves else 0
        if self.left_child:
            count += self.left_child.count_nodes_below(only_leaves)
        if self.right_child:
            count += self.right_child.count_nodes_below(only_leaves)
        return count

    def __str__(self):
        """
        Returns a string representation of the node and its subtree.
        """
        if self.is_root:
            s = "root"
        else:
            s = "-> node"
        node_description = f"{s} [feature={self.feature}, threshold={self.threshold}]"
        left_str = self.left_child_add_prefix(str(self.left_child)) if self.left_child else ""
        right_str = self.right_child_add_prefix(str(self.right_child)) if self.right_child else ""
        return f"{node_description}\n{left_str}{right_str}"

    def left_child_add_prefix(self, text):
        """
        Formats and returns text for the left child with appropriate tree graphics.
        """
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += "    |  " + x + "\n"
        return new_text.strip()

    def right_child_add_prefix(self, text):
        """
        Formats and returns text for the right child with appropriate tree graphics.
        """
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += "       " + x + "\n"
        return new_text.rstrip()

class Leaf(Node):
    """
    Represents a leaf node in a decision tree.

    Attributes:
        value (any): The value held by the leaf node.
        depth (int): The depth of the leaf node in the tree.
    """
    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """
        Returns the depth of the leaf node, as leaf nodes are the endpoints of the tree.
        """
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """
        Returns 1, as leaf nodes count as a single node.
        """
        return 1

    def __str__(self):
        return f"-> leaf [value={self.value}]"

class Decision_Tree():
    """
    Implements a decision tree for classification or regression.

    Attributes:
        max_depth (int): Maximum depth of the tree.
        min_pop (int): Minimum population required for a split.
        seed (int): Seed for the random number generator.
        split_criterion (str): Criterion used to choose the best split.
        root (Node): Root node of the tree.
    """
    def __init__(self, max_depth=10, min_pop=1, seed=0, split_criterion="random", root=None):
        self.rng = np.random.default_rng(seed)
        self.root = root if root else Node(is_root=True)
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion

    def depth(self):
        """
        Returns the maximum depth of the tree.
        """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """
        Counts the total nodes or only leaf nodes in the tree, depending on the flag.
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        return str(self.root)

