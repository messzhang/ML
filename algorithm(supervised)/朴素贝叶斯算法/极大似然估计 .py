###################################################
# Perceptron.py
# Author: zffy
# Date: 2024-05-22
# Description: Naive Bayes algorithm
###################################################
import numpy as np
from collections import defaultdict

class NaiveBayesMLE:
    def __init__(self):
        self.class_probs = {}
        self.feature_probs = defaultdict(dict)
        self.classes = []
    
    def fit(self, X, y):
        self.classes = np.unique(y)
        n_samples, n_features = X.shape
        for c in self.classes:
            X_c = X[y == c]
            self.class_probs[c] = X_c.shape[0] / n_samples#先验概率p(y=c)
            for j in range(n_features):
                feature_values, counts = np.unique(X_c[:, j], return_counts=True)
                self.feature_probs[c][j] = {v: count / X_c.shape[0] for v, count in zip(feature_values, counts)}#条件概率，P（X(j)=v|y=c）
    
    def predict(self, X):
        n_samples, n_features = X.shape
        preds = []
        for i in range(n_samples):
            posteriors = {}
            for c in self.classes:
                posterior = self.class_probs[c]
                for j in range(n_features):
                    if X[i, j] in self.feature_probs[c][j]:
                        posterior *= self.feature_probs[c][j][X[i, j]]
                    else:
                        posterior *= 1e-6  # Smoothing for unseen feature values
                posteriors[c] = posterior
            preds.append(max(posteriors, key=posteriors.get))
        return preds

# Example usage:
X = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
y = np.array([1, 1, 0, 0])
nb_mle = NaiveBayesMLE()
nb_mle.fit(X, y)
predictions = nb_mle.predict(X)
print(predictions)
