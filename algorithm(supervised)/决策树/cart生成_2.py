import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

# 生成简单的数据集
X = np.array([[0, 0], [1, 1], [1, 0], [0, 1], [0.5, 0.5], [1, 0.5], [0.5, 1], [1, 1.5], [1.5, 1], [1.5, 0.5]])
y = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 0])

# 初始化决策树分类器
clf = DecisionTreeClassifier()

# 拟合模型
clf.fit(X, y)

# 可视化决策树
plt.figure(figsize=(12, 8))
plot_tree(clf, filled=True, feature_names=['X1', 'X2'], class_names=['Class 0', 'Class 1'], rounded=True)
plt.title("CART Decision Tree on Simple Dataset")
plt.show()
