from decisionclassifier import DecisionTree
from joblib import Parallel, delayed
import numpy as np

# RandomForest class
class RandomForest:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, criterion='gini', max_features=None):
        """
        Initialize the Random Forest classifier.

        Parameters:
        n_estimators (int): The number of trees in the forest.
        max_depth (int): The maximum depth of the individual trees.
        min_samples_split (int): The minimum number of samples required to split an internal node.
        criterion (str): The function to measure the quality of a split ('gini' or 'entropy').
        max_features (int): The number of features to consider when looking for the best split.
        """
        self.n_estimators = n_estimators
        self.trees = []
        self.oob_samples_indices = []
        self.oob_predictions = {}
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.max_features = max_features
        self.oob_score_ = None  # Will hold the OOB score after fitting

    def fit(self, X, y):
        """
        Build a forest of trees from the training set (X, y).

        Parameters:
        X (pd.DataFrame): The training input samples.
        y (array-like): The target values (class labels).
        """
        self.trees = []
        self.oob_samples_indices = []
        n_samples = len(y)
        # Create bootstrap samples and train trees in parallel
        results = Parallel(n_jobs=-1)(
            delayed(self._train_tree)(X, y, n_samples) for _ in range(self.n_estimators)
        )
        # Unpack the results
        for tree, oob_indices in results:
            self.trees.append(tree)
            self.oob_samples_indices.append(oob_indices)
        # Compute OOB predictions
        self._compute_oob_score(X, y)

    def _train_tree(self, X, y, n_samples):
        """
        Train a single decision tree on a bootstrap sample of the dataset.

        Parameters:
        X (pd.DataFrame): The input samples.
        y (array-like): The target values.
        n_samples (int): The number of samples in the dataset.

        Returns:
        tuple: A tuple containing the trained tree and the indices of OOB samples.
        """
        # Bootstrap sampling
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        oob_indices = np.setdiff1d(np.arange(n_samples), indices)
        X_sample = X.iloc[indices]
        y_sample = y[indices]
        # Create and train a new tree
        tree = DecisionTree(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            criterion=self.criterion,
            max_features=self.max_features
        )
        tree.fit(X_sample, y_sample)
        return tree, oob_indices

    def _compute_oob_score(self, X, y):
        """
        Compute the Out-Of-Bag (OOB) score for the trained forest.

        Parameters:
        X (pd.DataFrame): The input samples.
        y (array-like): The target values.
        """
        n_samples = len(y)
        votes = np.zeros((n_samples,))  # Sum of votes for each sample
        vote_counts = np.zeros((n_samples,))  # Number of times each sample was OOB

        for tree, oob_indices in zip(self.trees, self.oob_samples_indices):
            if len(oob_indices) == 0:
                continue  # Skip if no OOB samples for this tree
            predictions = tree.predict(X.iloc[oob_indices])
            votes[oob_indices] += predictions
            vote_counts[oob_indices] += 1

        # Avoid division by zero
        valid_indices = vote_counts > 0
        oob_predictions = np.zeros(n_samples)
        oob_predictions[valid_indices] = (votes[valid_indices] / vote_counts[valid_indices]) >= 0.5

        # Calculate OOB score
        self.oob_score_ = np.mean(oob_predictions[valid_indices] == y[valid_indices])

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters:
        X (pd.DataFrame): The input samples.

        Returns:
        array: The predicted class labels.
        """
        # Collect predictions from all trees
        predictions = np.array([tree.predict(X) for tree in self.trees])
        # Majority vote
        majority_votes = np.apply_along_axis(
            lambda x: np.bincount(x, minlength=2).argmax(),
            axis=0,
            arr=predictions
        )
        return majority_votes