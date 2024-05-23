###################################################
# Perceptron.py
# Author: zffy
# Date: 2024-05-23
# Description: Perceptron algorithm version2
###################################################
import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.01,n_iter=1000):
        self.learning_rate = learning_rate
        self.n_iter=n_iter

    def fit(self, X, y):
        # 初始化权重向量和偏置
        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        self.errors_ = []
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                if(target*self.net_input(xi)<=0):
                    self.weights+=self.learning_rate*target*xi
                    self.bias+=self.learning_rate*target
                            # 统计错误数量
                    errors += 1
            self.errors_.append(errors)
        # for _ in range(self.n_iter):#计算每一个
        #     errors = 0
        #     for xi, target in zip(X, y):
        #         # 计算预测值
        #         update = self.learning_rate * (target - self.predict(xi))
        #         # 更新权重和偏置
        #         self.weights += update * xi
        #         self.bias += update
        #         # 统计错误数量
        #         errors += int(update != 0.0)
        #     self.errors_.append(errors)
        return self

    def net_input(self, X):
        """计算净输入"""
        return np.dot(X, self.weights) + self.bias

    def predict(self, X):
        """返回预测结果"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)

# 示例使用
if __name__ == "__main__":
    # 生成示例数据
    X = np.array([[2, 3], [4, 1], [1, 6], [6, 4], [1, 4], [7, 2]])
    y = np.array([1, 1, -1, -1, -1, 1])

    # 初始化感知机并训练模型
    perceptron = Perceptron(learning_rate=0.01,n_iter=100)
    perceptron.fit(X, y)

    # 打印训练后的权重和偏置
    print("Weights:", perceptron.weights)
    print("Bias:", perceptron.bias)

    # 预测
    predictions = perceptron.predict(X)
    print("Predictions:", predictions)

    #target
    print("target:",y)

    # 打印错误数量
    print("Errors:", perceptron.errors_)