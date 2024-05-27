import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree

# 生成样本数据
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
y = np.array([1.2, 3.4, 5.6, 7.8, 9.0])

# 初始化回归树模型
regressor = DecisionTreeRegressor()

# 拟合模型
regressor.fit(X, y)

# 可视化回归树
plt.figure(figsize=(12, 8))
plot_tree(regressor, filled=True, feature_names=['Feature 1', 'Feature 2'], rounded=True)
plt.title("Regression Tree")
plt.show()

# Output:

print(regressor.predict([[2, 6], [4, 8], [6, 10], [8, 12]]))

