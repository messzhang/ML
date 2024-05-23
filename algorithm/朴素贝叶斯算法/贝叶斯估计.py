import numpy as np
from collections import defaultdict

class NaiveBayesBayesian:
    def __init__(self, alpha=1):
        self.class_probs = {}
        self.feature_probs = defaultdict(dict)
        self.alpha = alpha
        self.classes = []
    
    def fit(self, X, y):
        self.classes = np.unique(y)
        n_samples, n_features = X.shape
        for c in self.classes:
            X_c = X[y == c]
            self.class_probs[c] = (X_c.shape[0] + self.alpha) / (n_samples + len(self.classes) * self.alpha)
            for j in range(n_features):
                feature_values, counts = np.unique(X_c[:, j], return_counts=True)
                total_count = sum(counts)
                self.feature_probs[c][j] = {v: (count + self.alpha) / (total_count + len(feature_values) * self.alpha) for v, count in zip(feature_values, counts)}
    
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
                        posterior *= self.alpha / (sum(self.feature_probs[c][j].values()) + len(self.feature_probs[c][j]) * self.alpha)
                posteriors[c] = posterior
            preds.append(max(posteriors, key=posteriors.get))
        return preds

# Example usage:
X = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
y = np.array([1, 1, 0, 0])
nb_bayesian = NaiveBayesBayesian()
nb_bayesian.fit(X, y)
predictions = nb_bayesian.predict(X)
print(predictions)
