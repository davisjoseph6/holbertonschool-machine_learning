#!/usr/bin/env python3

"""
This is the 9-random_forest module, relying on module
8-build_decision_tree.
"""
import numpy as np
from joblib import Parallel, delayed  # For parallel processing
Decision_Tree = __import__('8-build_decision_tree').Decision_Tree


class Random_Forest():
    """
    Random forest class, using Decision Trees.
    """
    def __init__(self, n_trees=100, max_depth=10, min_pop=1, seed=0):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.seed = seed
        self.trees = []  # Store tree objects directly for more flexibility
        self.explanatory = None  # To store the last trained features
        self.target = None       # To store the last trained targets

    def predict(self, explanatory):
        """
        Returns an array of the most frequent prediction for each tree in
        self.trees, based on the given explanatory variables.
        """
        if not self.trees:
            raise ValueError("The model has not been trained yet.")
        # Parallel computation of predictions
        predictions = np.array(Parallel(n_jobs=-1)(
            delayed(tree.predict)(explanatory) for tree in self.trees))

        # Calculate the mode (most frequent) prediction for each example
        mode_predictions = np.apply_along_axis(
            lambda x: np.bincount(x, minlength=np.max(x) + 1).argmax(),
            axis=0, arr=predictions)
        return mode_predictions

    def fit(self, explanatory, target, verbose=0):
        """
        Fits the random forest to the given training data.
        """
        self.explanatory = explanatory  # Store the training features
        self.target = target            # Store the training targets
        self.trees = [Decision_Tree(max_depth=self.max_depth,
                                    min_pop=self.min_pop,
                                    seed=self.seed + i)
                      for i in range(self.n_trees)]

        # Parallel training of decision trees
        trained_trees = Parallel(n_jobs=-1)(
            delayed(self._train_tree)(tree, explanatory, target)
            for tree in self.trees)

        self.trees = trained_trees  # Store trained trees

        if verbose == 1:
            depths = [tree.depth() for tree in self.trees]
            nodes = [tree.count_nodes() for tree in self.trees]
            leaves = [tree.count_nodes(only_leaves=True) for tree in self.trees]
            accuracies = [tree.accuracy(explanatory, target) for tree in self.trees]
            print(f"""  Training finished.
    - Mean depth                     : {np.mean(depths)}
    - Mean number of nodes           : {np.mean(nodes)}
    - Mean number of leaves          : {np.mean(leaves)}
    - Mean accuracy on training data : {np.mean(accuracies)}
    - Accuracy of the forest on td   : {self.accuracy(explanatory, target)}""")

    def _train_tree(self, tree, explanatory, target):
        tree.fit(explanatory, target)
        return tree

    def accuracy(self, test_explanatory, test_target):
        """
        Calculates the accuracy of the random forest on the given test data.
        """
        predictions = self.predict(test_explanatory)
        return np.sum(predictions == test_target) / len(test_target)

