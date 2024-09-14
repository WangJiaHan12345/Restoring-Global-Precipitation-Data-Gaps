import numpy as np

# 定义一个列表，存储原始数据
data = [7.98, 6.22, 3.29, 3.86]

# 计算数据的平均值和标准差
mean = np.mean(data)
std = np.std(data)

# 对数据进行标准化处理
normalized_data = (data - mean) / std

# 输出标准化后的数据
print("原始数据：", data)
print("标准化后的数据：", normalized_data)