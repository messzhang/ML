import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree

# 加载示例数据集
iris = load_iris()
X, y = iris.data, iris.target

# 初始化决策树分类器
clf = DecisionTreeClassifier()

# 拟合模型
clf.fit(X, y)

# 可视化决策树
plt.figure(figsize=(20, 10))
plot_tree(clf, filled=True, feature_names=iris.feature_names, class_names=iris.target_names, rounded=True)
plt.title("CART Decision Tree")
plt.show()
