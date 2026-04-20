import numpy as np

# 1. 定义异或问题 (XOR) 的绝对物理坐标与标签
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
Y = np.array([[0], [1], [1], [0]]) # 保持二维列向量形态

# 2. 强行固化锚点 (Centers)
# 极其冷酷地将 2 个锚点死死钉在两个正例上
centers = np.array([
    [0, 1],
    [1, 0]
])

# 3. 设定高斯核衰减参数 beta
# 这里取 beta = 2.0，可以让远处的点衰减得更狠，空间撕裂效果更明显
beta = 2.0

# 4. 空间扭曲：构建隐藏层特征矩阵 G (形状: 4 x 2)
n_samples = X.shape[0]
k_centers = centers.shape[0]
G = np.zeros((n_samples, k_centers))

for i in range(n_samples):
    for j in range(k_centers):
        # 计算欧氏距离的平方并进行高斯衰减
        distance_sq = np.sum((X[i] - centers[j]) ** 2)
        G[i, j] = np.exp(-beta * distance_sq)

print("=== 空间跃迁后的新坐标 (隐藏层 G 矩阵) ===")
# 你将在这里极其直观地看到 (0,0) 和 (1,1) 的坐标发生了物理坍缩与重叠
print(np.round(G, 4))
print("-" * 40)

# 5. 代数坍缩：使用伪逆矩阵一步求解输出层权重
# 偏置吸收：在 G 矩阵右侧强行拼接一列全 1，维度变为 4 x 3
G_with_bias = np.hstack((G, np.ones((n_samples, 1))))

# 求解 W_out (包含 2 个权重和 1 个偏置)，维度为 3 x 1
W_out = np.dot(np.linalg.pinv(G_with_bias), Y)

print("=== 绝对线性超平面的解析解 ===")
print(f"锚点 1 权重 : {W_out[0][0]:.4f}")
print(f"锚点 2 权重 : {W_out[1][0]:.4f}")
print(f"全局偏置 b  : {W_out[2][0]:.4f}")
print("-" * 40)

# 6. 终局验证：用解出来的超平面去切割映射后的空间
predictions = np.dot(G_with_bias, W_out)

print("=== 最终网络物理输出对撞 ===")
for i in range(n_samples):
    print(f"原始输入: {X[i]} -> 真实标签: {Y[i][0]} | 网络预测: {predictions[i][0]:.4f}")