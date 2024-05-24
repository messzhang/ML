import numpy as np
from collections import Counter
from math import log2

class DTNode:
    def __init__(self, feature=None, value=None, result=None, left=None, right=None):
        self.feature = feature
        self.value = value
        self.result = result
        self.left = left
        self.right = right

class DecisionTreeID3:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_samples_per_class = [np.sum(y == i) for i in np.unique(y)]
        most_common_class = np.argmax(n_samples_per_class)
        
        if len(np.unique(y)) == 1 or depth == self.max_depth:
            return DTNode(result=most_common_class)
        
        feature, value = self._best_split(X, y)
        left_indices = X[:, feature] <= value
        right_indices = X[:, feature] > value
        left = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
        right = self._grow_tree(X[right_indices], y[right_indices], depth + 1)
        return DTNode(feature, value, left=left, right=right)

    def _best_split(self, X, y):
        n_samples, n_features = X.shape
        best_info_gain = -float('inf')
        best_feature, best_value = None, None
        
        for feature in range(n_features):
            feature_values = np.unique(X[:, feature])
            for value in feature_values:
                left_indices = X[:, feature] <= value
                right_indices = X[:, feature] > value
                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue
                info_gain = self._information_gain(y, y[left_indices], y[right_indices])
                if info_gain > best_info_gain:
                    best_info_gain = info_gain
                    best_feature = feature
                    best_value = value
        return best_feature, best_value
    
    def _entropy(self, y):
        counts = np.bincount(y)
        probabilities = counts / len(y)
        return -np.sum([p * log2(p) for p in probabilities if p > 0])
    
    def _information_gain(self, y, y_left, y_right):
        p = len(y_left) / len(y)
        return self._entropy(y) - p * self._entropy(y_left) - (1 - p) * self._entropy(y_right)
    
    def predict(self, X):
        return [self._predict(x) for x in X]
    
    def _predict(self, x, node=None):
        if node is None:
            node = self.root
        if node.result is not None:
            return node.result
        if x[node.feature] <= node.value:
            return self._predict(x, node.left)
        else:
            return self._predict(x, node.right)

# Example usage:
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the decision tree
dt = DecisionTreeID3(max_depth=3)
dt.fit(X_train, y_train)

# Predict on the test set
predictions = dt.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.2f}")
