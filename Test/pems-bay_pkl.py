import numpy as np
import pandas as pd
import pickle
import os

df = pd.read_pickle("C:\\Code\\BasicTS-master\\datasets\\PEMS-BAY\\adj_mx.pkl")
a = df[2]

# 初始化新的对称无向图邻接矩阵 b
b = np.zeros_like(a)

# 遍历上三角元素，并设置对称连接
for i in range(a.shape[0]):
    for j in range(i + 1, a.shape[1]):
        if a[i][j] != 0 or a[j][i] != 0:
            b[i][j] = 1
            b[j][i] = 1
# 指定保存位置和文件名
save_path = r'C:\Code\BasicTS-master\datasets\PEMS-BAY\adj_mx_process.pkl'

# 确保目标文件夹存在
os.makedirs(os.path.dirname(save_path), exist_ok=True)

# 使用 pickle 保存数组到指定的 pkl 文件
with open(save_path, 'wb') as f:
    pickle.dump(b, f)

print(f"数组已保存到 {save_path}")
