import numpy as np
from joblib import Parallel, delayed
import pydotplus
from collections import deque
import matplotlib.pyplot as plt

# Helper functions for parallel processing

def calculate_loss(y, criterion):
    """
    Calculate the impurity (loss) of a node based on the specified criterion.

    Parameters:
    y (array-like): Target values at the node.
    criterion (str): The impurity criterion ('gini' or 'entropy').

    Returns:
    float: The calculated impurity value.
    """
    counts = np.bincount(y)
    probabilities = counts / len(y)
    if criterion == 'gini':
        return 1 - np.sum(probabilities ** 2)
    elif criterion == 'entropy':
        return -np.sum([p * np.log2(p) if p > 0 else 0 for p in probabilities])
    else:
        raise ValueError("Invalid criterion specified.")

def information_gain(parent_loss, left_y, right_y, criterion):
    """
    Calculate the information gain from a potential split.

    Parameters:
    parent_loss (float): Impurity of the parent node.
    left_y (array-like): Target values of the left child node.
    right_y (array-like): Target values of the right child node.
    criterion (str): The impurity criterion ('gini' or 'entropy').

    Returns:
    float: The information gain resulting from the split.
    """
    weight_left = len(left_y) / (len(left_y) + len(right_y))
    weight_right = 1 - weight_left
    gain = parent_loss
    gain -= weight_left * calculate_loss(left_y, criterion)
    gain -= weight_right * calculate_loss(right_y, criterion)
    return gain

def find_best_split_for_feature(feature, X_feature, y, criterion):
    """
    Find the best threshold or category to split on for a given feature.

    Parameters:
    feature (str): The name of the feature.
    X_feature (pd.Series): Feature values.
    y (array-like): Target values.
    criterion (str): The impurity criterion ('gini' or 'entropy').

    Returns:
    tuple: A tuple containing the feature name, best information gain, and best threshold.
    """
    parent_loss = calculate_loss(y, criterion)
    best_gain = -1
    best_threshold = None

    # Handle categorical features
    if X_feature.dtype == 'object':
        categories = X_feature.unique()
        if len(categories) == 1:
            return feature, best_gain, best_threshold

        # Consider one-vs-rest splits
        for category in categories:
            left_indices = X_feature == category
            right_indices = ~left_indices
            if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
                continue

            gain = information_gain(
                parent_loss, y[left_indices], y[right_indices], criterion
            )
            if gain > best_gain:
                best_gain = gain
                best_threshold = set([category])
    else:
        # Numerical features
        thresholds = np.unique(X_feature)
        thresholds.sort()

        # Consider midpoints between consecutive values
        midpoints = (thresholds[:-1] + thresholds[1:]) / 2

        for threshold in midpoints:
            left_indices = X_feature <= threshold
            right_indices = X_feature > threshold
            if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
                continue

            gain = information_gain(
                parent_loss, y[left_indices], y[right_indices], criterion
            )
            if gain > best_gain:
                best_gain = gain
                best_threshold = threshold

    return feature, best_gain, best_threshold

def train_test_split_custom(X, y, test_size=0.2, random_state=None):
    """
    Split the dataset into training and testing sets.

    Parameters:
    X (pd.DataFrame): The input features.
    y (array-like): The target values.
    test_size (float): The proportion of the dataset to include in the test split.
    random_state (int): Seed for the random number generator.

    Returns:
    tuple: X_train, X_test, y_train, y_test
    """
    np.random.seed(random_state)
    indices = np.arange(len(y))
    np.random.shuffle(indices)
    split_index = int(len(y) * (1 - test_size))
    train_indices = indices[:split_index]
    test_indices = indices[split_index:]
    X_train = X.iloc[train_indices]
    X_test = X.iloc[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    return X_train, X_test, y_train, y_test

def cross_validate(model_class, X, y, params, cv=3):
    """
    Perform cross-validation for model evaluation.

    Parameters:
    model_class (class): The model class to instantiate.
    X (pd.DataFrame): The input features.
    y (array-like): The target values.
    params (dict): Parameters to initialize the model.
    cv (int): Number of cross-validation folds.

    Returns:
    float: The mean accuracy across all folds.
    """
    n_samples = len(y)
    fold_size = n_samples // cv
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    scores = []

    for fold in range(cv):
        start = fold * fold_size
        end = start + fold_size if fold != cv - 1 else n_samples
        val_indices = indices[start:end]
        train_indices = np.concatenate((indices[:start], indices[end:]))

        X_train_cv = X.iloc[train_indices]
        y_train_cv = y[train_indices]
        X_val_cv = X.iloc[val_indices]
        y_val_cv = y[val_indices]

        model = model_class(**params)
        model.fit(X_train_cv, y_train_cv)
        predictions = model.predict(X_val_cv)
        accuracy = np.mean(predictions == y_val_cv)
        scores.append(accuracy)

    return np.mean(scores)

def hyperparameter_tuning(model_class, X, y, param_grid, cv=3):
    """
    Perform hyperparameter tuning using grid search.

    Parameters:
    model_class (class): The model class to instantiate.
    X (pd.DataFrame): The input features.
    y (array-like): The target values.
    param_grid (dict): Dictionary with parameters names as keys and lists of parameter settings to try.
    cv (int): Number of cross-validation folds.

    Returns:
    dict: The best parameters found during tuning.
    """
    from itertools import product

    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in product(*values)]

    def evaluate_params(params):
        score = cross_validate(model_class, X, y, params, cv=cv)
        print(f"Params: {params}, Score: {score}")
        return (score, params)

    results = Parallel(n_jobs=-1)(
        delayed(evaluate_params)(params) for params in param_combinations
    )

    # Find the best hyperparameters
    best_score, best_params = max(results, key=lambda x: x[0])

    print("\nBest Parameters:")
    print(best_params)
    print(f"Best Cross-Validation Score: {best_score}")

    return best_params

# Compute confusion matrix function

def compute_confusion_matrix(y_true, y_pred):
    """
    Compute the confusion matrix for binary classification.

    Parameters:
    y_true (array-like): True class labels.
    y_pred (array-like): Predicted class labels.

    Returns:
    np.ndarray: A 2x2 confusion matrix.
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))  # True Positives
    tn = np.sum((y_true == 0) & (y_pred == 0))  # True Negatives
    fp = np.sum((y_true == 0) & (y_pred == 1))  # False Positives
    fn = np.sum((y_true == 1) & (y_pred == 0))  # False Negatives
    return np.array([[tn, fp],
                     [fn, tp]])

def compute_metrics(y_true, y_pred):
    """
    Compute precision, recall, and F1 score for binary classification.

    Parameters:
    y_true (array-like): True class labels.
    y_pred (array-like): Predicted class labels.

    Returns:
    tuple: Precision, recall, and F1 score.
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))  # True Positives
    fp = np.sum((y_true == 0) & (y_pred == 1))  # False Positives
    fn = np.sum((y_true == 1) & (y_pred == 0))  # False Negatives
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    return precision, recall, f1_score

# Plot confusion matrix function

def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    Plot the confusion matrix using Matplotlib.

    Parameters:
    cm (np.ndarray): Confusion matrix to be plotted.
    classes (list): List of class names.
    title (str): Title of the plot.
    cmap (Colormap): Colormap instance for the plot.
    """
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    # Format the counts in the matrix
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            plt.text(j, i, format(count, 'd'),
                     horizontalalignment='center',
                     color='white' if count > thresh else 'black')

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

def export_tree_to_graphviz(node):
    """
    Export a decision tree to Graphviz format for visualization.

    Parameters:
    node (Node): The root node of the decision tree.

    Returns:
    graph: A pydotplus graph object representing the decision tree.
    """
    dot_data = 'digraph Tree {\n'
    node_id = 0
    node_ids = {id(node): node_id}
    queue = deque()
    queue.append(node)

    while queue:
        current_node = queue.popleft()
        current_id = node_ids[id(current_node)]

        # Define the label for the current node
        if current_node.is_leaf:
            if current_node.prediction == 1:
                label = 'Poisonous'
                dot_data += f'  {current_id} [label="{label}", shape="ellipse", style="filled", color="plum", width=1.5, height=1.0];\n'
            else:
                label = 'Edible'
                dot_data += f'  {current_id} [label="{label}", shape="ellipse", style="filled", color="lightgreen", width=1.5, height=1.0];\n'
        else:
            if isinstance(current_node.threshold, set):
                threshold = ', '.join(str(t) for t in current_node.threshold)
                label = f'{current_node.feature} in [{threshold}]'
            else:
                threshold = current_node.threshold.round(4)
                label = f'{current_node.feature} ≤ {threshold}'
            dot_data += f'  {current_id} [label="{label}", shape="ellipse", style="filled", color="paleturquoise", width=1.5, height=1.0];\n'

            # Left child (True branch)
            left_id = len(node_ids)
            node_ids[id(current_node.left)] = left_id
            queue.append(current_node.left)
            dot_data += f'  {current_id} -> {left_id} [label="Left"];\n'

            # Right child (False branch)
            right_id = len(node_ids)
            node_ids[id(current_node.right)] = right_id
            queue.append(current_node.right)
            dot_data += f'  {current_id} -> {right_id} [label="Right"];\n'

    dot_data += '}'
    graph = pydotplus.graph_from_dot_data(dot_data)
    return graph
