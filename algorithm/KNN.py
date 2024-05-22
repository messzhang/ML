###################################################
# Perceptron.py
# Author: zffy
# Date: 2024-05-22
# Description: KNN algorithm
###################################################
import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k=3):
        self.k = k
    
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    
    def predict(self, X_test):
        predictions = [self._predict(x) for x in X_test]
        return np.array(predictions)
    
    def _predict(self, x):
        # 计算所有训练样本与测试样本 x 之间的欧氏距离
        distances = [np.linalg.norm(x - x_train) for x_train in self.X_train]
        # 获取距离最近的 k 个样本的标签
        k_indices = np.argsort(distances)[:self.k]#用于返回数组元素从小到大排序的索引
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # 返回出现次数最多的标签
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

# 创建示例数据
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 加载鸢尾花数据集
# 鸢尾花数据集（Iris Dataset）是机器学习和统计学中常用的经典数据集之一。
# 它包含 150 个样本，每个样本有 4 个特征和一个标签。这些特征是花萼长度、花萼宽度、花瓣长度和花瓣宽度，标签则是鸢尾花的三种类型之一：Setosa、Versicolor 和 Virginica。
iris = load_iris()
X, y = iris.data, iris.target

# 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)

# 创建KNN分类器实例
knn = KNN(k=3)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)

# 评估模型
accuracy = np.sum(predictions == y_test) / len(y_test)
print(f'Accuracy: {accuracy:.4f}')

# 显示预测结果和真实标签
print('Predicted labels:', predictions)
print('True labels:', y_test)