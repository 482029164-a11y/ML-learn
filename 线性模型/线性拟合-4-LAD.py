import numpy as np
# 需要记得三个点：1.涉及到求逆操作，一定要扩展特征值
# 2.查找，标签为1的所有X_0
# 3.求平均值不能简单mean,要点名方向
def generate_lda_data():
    np.random.seed(42)
    m = 200
    cov = np.array([[3.0, 2.5], [2.5, 3.0]])
    X_0 = np.random.multivariate_normal([1, 1], cov, m)
    y_0 = np.zeros((m, 1))
    X_1 = np.random.multivariate_normal([4, 4], cov, m)
    y_1 = np.ones((m, 1))
    X = np.vstack((X_0, X_1))
    y = np.vstack((y_0, y_1))
    return X, y


class LinearDiscriminantAnalysisRaw:
    def __init__(self, lambda_reg=1e-4):
        self.lambda_reg = lambda_reg
        self.w = None
        self.threshold = 0

    def fit(self, X, y):
        # 1. 物理分离
        X_0 = X[y.flatten() == 0]
        X_1 = X[y.flatten() == 1]  # ⚠️ 修正：提取标签 1

        # 2. 寻找重心 (均值列向量)
        # ⚠️ 修正：指定 axis=0 顺列求均值，并用 reshape 强制锁定为 (d, 1) 的列向量
        mu_0 = np.mean(X_0, axis=0).reshape(-1, 1)
        mu_1 = np.mean(X_1, axis=0).reshape(-1, 1)

        # 3. 构造类内散度矩阵 S_w (维度必须是 d x d)
        # ⚠️ 修正：正确的中心化广播机制 (m, d) - (1, d)
        X_0_c = X_0 - mu_0.T
        X_1_c = X_1 - mu_1.T

        # ⚠️ 修正：(d, m) @ (m, d) -> (d, d) 才是正确的协方差散度阵
        S_0 = X_0_c.T @ X_0_c
        S_1 = X_1_c.T @ X_1_c
        S_w = S_0 + S_1

        # 4. 实施代数打击
        d = X.shape[1]
        I = np.eye(d)
        # 此时 S_w 和 I 的维度都是 (2, 2)，相加绝对安全
        S_w_safe = S_w + self.lambda_reg * I

        try:
            self.w = np.linalg.inv(S_w_safe) @ (mu_1 - mu_0)
        except np.linalg.LinAlgError:
            self.w = np.linalg.pinv(S_w_safe) @ (mu_1 - mu_0)

        # 5. 确立绝对切割点
        mu_0_proj = mu_0.T @ self.w
        mu_1_proj = mu_1.T @ self.w
        self.threshold = (mu_0_proj + mu_1_proj) / 2.0

    def predict(self, X):
        if self.w is None:
            raise ValueError("模型尚未训练！")

        scores = X @ self.w
        preds = (scores >= self.threshold).astype(int)
        return preds


if __name__ == "__main__":
    X_train, y_train = generate_lda_data()

    lda = LinearDiscriminantAnalysisRaw(lambda_reg=1e-4)
    lda.fit(X_train, y_train)

    preds = lda.predict(X_train)
    accuracy = np.mean(preds == y_train)

    print("最优投影向量 w:\n", lda.w)
    print("一维切割阈值 threshold:", float(lda.threshold))
    print(f"LDA 纯矩阵化模型准确率: {accuracy * 100:.2f}%")