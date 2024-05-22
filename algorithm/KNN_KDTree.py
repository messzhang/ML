###################################################
# Perceptron.py
# Author: zffy
# Date: 2024-05-22
# Description: KNN algorithm
###################################################
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KDTree
from collections import Counter

class KNN_KDTree:
    def __init__(self, k=3):
        self.k = k
    
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.tree = KDTree(X_train)
    
    def predict(self, X_test):
        predictions = [self._predict(x) for x in X_test]
        return np.array(predictions)
    
    def _predict(self, x):
        # 查询最近的 k 个邻居
        dist, idx = self.tree.query([x], k=self.k)#返回的是一个二维数组，第一个维度是距离，第二个维度是索引
        # 获取这 k 个邻居的标签
        k_nearest_labels = [self.y_train[i] for i in idx[0]]
        # 返回出现次数最多的标签
        most_common = Counter(k_nearest_labels).most_common(1)#返回一个列表，列表中的元素是元组，元组的第一个元素是标签，第二个元素是标签出现的次数,most_common(1)表示返回出现次数最多的一个标签
        return most_common[0][0]


# 加载鸢尾花数据集
iris = load_iris()
X, y = iris.data, iris.target

# 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)


# 创建KNN分类器实例
knn_kdtree = KNN_KDTree(k=3)
knn_kdtree.fit(X_train, y_train)
predictions = knn_kdtree.predict(X_test)

# 评估模型
accuracy = np.sum(predictions == y_test) / len(y_test)
print(f'Accuracy: {accuracy:.4f}')

# 显示预测结果和真实标签
print('Predicted labels:', predictions)
print('True labels:', y_test)
