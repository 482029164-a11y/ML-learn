import numpy as np


class RBFNetwork:
    def __init__(self, k_centers):
        # k_centers 决定了我们在相空间中撒下几个“锚点”
        self.k = k_centers
        self.centers = None
        self.beta = None
        self.W = None

    def _rbf_kernel(self, x, c, beta):
        # 绝对物理公式：计算点 x 与锚点 c 的欧氏距离的平方，并进行高斯衰减
        distance_sq = np.sum((x-c)**2)
        return np.exp(-beta * distance_sq)

    def fit(self, X, y):
        n=X.shape[0]

        # ==========================================
        # 阶段一：空间锚点固化 (撒网)
        # ==========================================
        np.random.seed(42)
        # 极其暴力的启发式：直接从训练集中随机抽取 k 个瓜作为锚点
        num_center=np.random.choice(n, self.k, replace=False)
        self.centers=X[num_center]
        # 启发式确定高斯核的衰减参数 beta
        # 物理规则：beta = 1 / (2 * sigma^2)。
        # sigma 取锚点间的最大距离除以 sqrt(2k)，保证各个高斯泡泡能适度重叠
        d_max=0
        for i in range(self.k):
            for j in range(self.k):
                d=np.linalg.norm(self.centers[i]-self.centers[j])
                if d>d_max:
                    d_max=d
        sigma=d_max/np.sqrt(2*self.k)
        self.beta=1/(2*(sigma**2))

        # ==========================================
        # 阶段二：非线性映射特征矩阵构建
        # ==========================================
        # G 矩阵的维度是 (样本数 n, 锚点数 k)
        G=np.zeros((n,self.k))
        for i in range(n):
            for j in range(self.k):
                G[i][j]=self._rbf_kernel(X[i], self.centers[j], self.beta)

        # ==========================================
        # 阶段三：绝对线性的代数坍缩 (伪逆求解)
        # ==========================================
        # G * W = y  =>  W = pinv(G) * y
        # 彻底消灭 for 循环和学习率，一击必杀求出最优权重
        self.W=np.linalg.pinv(G)@y

    def predict(self, X):
        # 推理阶段：重复矩阵 G 的构建过程
        n_samples = X.shape[0]
        G = np.zeros((n_samples, self.k))
        for i in range(n_samples):
            for j in range(self.k):
                G[i, j] = self._rbf_kernel(X[i], self.centers[j], self.beta)

        # 线性矩阵相乘，得出最终连续输出
        return np.dot(G, self.W)


# --- 物理演化测试入口 ---
if __name__ == "__main__":
    # 西瓜数据集 3.0 alpha (密度, 含糖率)
    X_data = np.array([
        [0.697, 0.460], [0.774, 0.376], [0.634, 0.264], [0.608, 0.318],
        [0.556, 0.215], [0.403, 0.237], [0.481, 0.149], [0.437, 0.211],
        [0.666, 0.091], [0.243, 0.267], [0.245, 0.057], [0.343, 0.099],
        [0.639, 0.161], [0.657, 0.198], [0.360, 0.370], [0.593, 0.042],
        [0.719, 0.103]
    ])

    # RBF 往往输出连续值，我们直接使用一维标签即可
    y_data = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).T

    # 实例化 RBF 网络，设置 5 个高斯基锚点
    rbf_net = RBFNetwork(k_centers=5)

    # 瞬间收敛，没有 epoch，没有 learning_rate
    rbf_net.fit(X_data, y_data)

    # 执行预测并计算均方误差
    predictions = rbf_net.predict(X_data)
    mse_loss = np.mean((predictions - y_data) ** 2)

    print(f"=== RBF 网络代数坍缩终局评估 ===")
    print(f"设定锚点数 (隐层节点): {rbf_net.k}")
    print(f"伪逆矩阵一步求解残差 (MSE): {mse_loss:.6f}")

    # 打印前 3 个样本的连续预测值，对比真实值 1
    print("\n前3个好瓜的绝对预测打分:")
    for i in range(3):
        print(f"瓜 {i} 预测分: {predictions[i][0]:.4f} (目标 1.0)")