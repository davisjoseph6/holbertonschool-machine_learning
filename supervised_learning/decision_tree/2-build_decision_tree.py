#!/usr/bin/env python3

"""
Decision Tree Components
Includes classes for nodes (both decision and leaf nodes) and the
decision tree itself
"""
import numpy as np

class Node:
    """
    Represents a node in a decision tree.
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
        if self.is_leaf:
            return self.depth
        depths = [self.depth]
        if self.left_child is not None:
            depths.append(self.left_child.max_depth_below())
        if self.right_child is not None:
            depths.append(self.right_child.max_depth_below())
        return max(depths)

    def count_nodes_below(self, only_leaves=False):
        if self.is_leaf:
            return 1 if only_leaves else 1
        count = 0 if only_leaves else 1
        if self.left_child:
            count += self.left_child.count_nodes_below(only_leaves)
        if self.right_child:
            count += self.right_child.count_nodes_below(only_leaves)
        return count

    def __str__(self):
        node_label = f"node [feature={self.feature}, threshold={self.threshold}]"
        if self.is_root:
            node_label = "root " + node_label
        children_str = ""
        if self.left_child:
            children_str += "\n    +---> " + str(self.left_child).replace("\n", "\n    |     ")
        if self.right_child:
            children_str += "\n    +---> " + str(self.right_child).replace("\n", "\n    |     ")
        return f"{node_label}{children_str}"

class Leaf(Node):
    """
    Represents a leaf node in a decision tree.
    """
    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def __str__(self):
        return f"leaf [value={self.value}]"

class Decision_Tree():
    """
    Implements a decision tree.
    """
    def __init__(self, max_depth=10, min_pop=1, seed=0, split_criterion="random", root=None):
        self.rng = np.random.default_rng(seed)
        self.root = root if root else Node(is_root=True)
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion

    def depth(self):
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        return self.root.count_nodes_below(only_leaves)

    def __str__(self):
        return str(self.root)


