import numpy as np
from joblib import Parallel, delayed
from helpfunction import calculate_loss, find_best_split_for_feature

# Node class remains the same
class Node:
    def __init__(self, is_leaf=False, prediction=None, feature=None, threshold=None, left=None, right=None):
        """
        Initialize a node in the decision tree.

        Parameters:
        is_leaf (bool): Indicates if the node is a leaf node.
        prediction: The predicted class if this is a leaf node.
        feature: The feature to split on.
        threshold: The threshold value for the feature to split on.
        left (Node): The left child node.
        right (Node): The right child node.
        """
        self.is_leaf = is_leaf
        self.prediction = prediction
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right

    def decide(self, x):
        """
        Recursively traverse the tree to make a prediction for a single sample.

        Parameters:
        x (pd.Series): A single sample from the dataset.

        Returns:
        The predicted class label.
        """
        if self.is_leaf:
            return self.prediction
        else:
            feature_value = x[self.feature]
            if isinstance(self.threshold, set):
                # Categorical feature
                if feature_value in self.threshold:
                    return self.left.decide(x)
                else:
                    return self.right.decide(x)
            else:
                # Numerical feature (if any)
                if feature_value <= self.threshold:
                    return self.left.decide(x)
                else:
                    return self.right.decide(x)
                    
# Modified Decision Tree class with max_features
class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, criterion='gini', max_features=None):
        """
        Initialize the Decision Tree classifier.

        Parameters:
        max_depth (int): The maximum depth of the tree.
        min_samples_split (int): The minimum number of samples required to split an internal node.
        criterion (str): The function to measure the quality of a split ('gini' or 'entropy').
        max_features (int): The number of features to consider when looking for the best split.
        """
        self.root = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.max_features = max_features

    def fit(self, X, y):
        """
        Build a decision tree classifier from the training set (X, y).

        Parameters:
        X (pd.DataFrame): The training input samples.
        y (array-like): The target values (class labels).
        """
        self.features = X.columns.tolist()
        self.n_features_ = len(self.features)
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        """
        Recursively grow the decision tree.

        Parameters:
        X (pd.DataFrame): The input samples.
        y (array-like): The target values.
        depth (int): The current depth of the tree.

        Returns:
        Node: The root node of the subtree.
        """
        num_samples = y.shape[0]
        num_labels = len(np.unique(y))

        if (self.max_depth is not None and depth >= self.max_depth) or num_labels == 1 or num_samples < self.min_samples_split:
            leaf_prediction = self._most_common_label(y)
            return Node(is_leaf=True, prediction=leaf_prediction)

        best_feature, best_gain, best_threshold = self._best_split(X, y)
        if best_feature is None or best_gain <= 0:
            leaf_prediction = self._most_common_label(y)
            return Node(is_leaf=True, prediction=leaf_prediction)

        # Split the dataset
        X_feature = X[best_feature]
        if isinstance(best_threshold, set):
            left_indices = X_feature.isin(best_threshold)
            right_indices = ~left_indices
        else:
            left_indices = X_feature <= best_threshold
            right_indices = X_feature > best_threshold

        left = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
        right = self._grow_tree(X[right_indices], y[right_indices], depth + 1)
        return Node(feature=best_feature, threshold=best_threshold, left=left, right=right)

    def _best_split(self, X, y):
        """
        Find the best feature and threshold to split on.

        Parameters:
        X (pd.DataFrame): The input samples.
        y (array-like): The target values.

        Returns:
        best_feature: The best feature to split on.
        best_gain: The information gain from the best split.
        best_threshold: The threshold value for the best split.
        """
        parent_loss = calculate_loss(y, self.criterion)
        # Select random subset of features if max_features is set
        if self.max_features is None or self.max_features >= len(self.features):
            features = self.features
        else:
            features = np.random.choice(self.features, self.max_features, replace=False)
        # Prepare arguments for parallel processing
        args_list = [
            (feature, X[feature], y, self.criterion) for feature in features
        ]

        # Use joblib's Parallel and delayed functions for parallel processing
        results = Parallel(n_jobs=-1)(
            delayed(find_best_split_for_feature)(feature, X_feature, y, self.criterion)
            for feature, X_feature, y, self.criterion in args_list
        )

        # Find the best feature to split on
        best_gain = -1
        best_feature = None
        best_threshold = None
        for feature, gain, threshold in results:
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = threshold

        return best_feature, best_gain, best_threshold

    def _most_common_label(self, y):
        """
        Find the most common class label in the target values.

        Parameters:
        y (array-like): The target values.

        Returns:
        The most common class label.
        """
        counts = np.bincount(y)
        return np.argmax(counts)

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters:
        X (pd.DataFrame): The input samples.

        Returns:
        array: The predicted class labels.
        """
        return X.apply(self._predict_row, axis=1).values

    def _predict_row(self, x):
        """
        Predict class label for a single sample.

        Parameters:
        x (pd.Series): A single input sample.

        Returns:
        The predicted class label.
        """
        return self.root.decide(x)
