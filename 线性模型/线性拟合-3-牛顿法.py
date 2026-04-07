import numpy as np
# 本例涉及四个易错点：1.牛顿法也需要迭代，不过迭代次数没有这么多2.利用numpy的广播机制可以降低对角矩阵的内存占用率
def pro_data():
    m1 = 150
    m2 = 300
    x1 = np.random.randn(m1, 2) + np.array([2, 2])
    x2 = np.random.randn(m1, 2) + np.array([-2, -2])
    y2 = np.ones((m1, 1))
    y1 = np.zeros((m1, 1))
    x = np.vstack((x1, x2))
    y = np.vstack((y1, y2))
    permutation = np.random.permutation(m2)
    x = x[permutation]
    y = y[permutation]
    return x, y


class LogisticRegressionNewton:
    def __init__(self, max_iter=10):
        # 牛顿法极度自信，不需要学习率 (lr)，只需要较少的迭代次数
        self.max_iter = max_iter
        self.w = None

    def sigmoid(self, z):
        z_safe = np.clip(z, -250, 250)
        return 1.0 / (1.0 + np.exp(-z_safe))

    def fit(self, x, y):
        m, n = x.shape
        i1 = np.ones((m, 1))
        x_aug = np.hstack((i1, x))

        # 建议初始化为 0，防止极端值导致早期梯度消失
        self.w = np.zeros((n + 1, 1))

        # ⚠️ 修正 4：牛顿法依然需要几轮迭代
        for i in range(self.max_iter):
            z = x_aug @ self.w
            p = self.sigmoid(z)
            erro = p - y
            # ⚠️ 修正 1：计算正确的方差向量 v (维度 m x 1)
            v = p * (1.0 - p)
            # 计算一阶导数（梯度）
            g = x_aug.T @ erro
            # ⚠️ 修正 2 & 降维：利用广播机制计算海森矩阵，绝对不开辟 m x m 的对角阵内存
            # (v * x_aug) 等价于 D @ X。整体等价于 X^T D X
            H = x_aug.T @ (v * x_aug)
            # ⚠️ 修正 3：必须给海森矩阵求逆，并加入 pinv 物理防崩塌兜底
            try:
                # 更新公式： w = w - H^(-1) * g
                self.w = self.w - np.linalg.inv(H) @ g
            except np.linalg.LinAlgError:
                self.w = self.w - np.linalg.pinv(H) @ g

    def predict(self, x):
        if self.w is None:
            print("请先调用fit函数")
            return
        m = x.shape[0]
        i1 = np.ones((m, 1))
        x_aug = np.hstack((i1, x))
        # ⚠️ 预测必须经过 Sigmoid 压缩，并做硬判决
        probs = self.sigmoid(x_aug @ self.w)
        return (probs >= 0.5).astype(int)
# astype(int)把所有的true变成1，这样方便之后准确率的计算
# probs >= 0.5利用了numpy的广播特性，会自动把标量变成和probs一样的矩阵，结果也会变成bool矩阵
if __name__ == "__main__":
    x_a, y_a = pro_data()
    l1 = LogisticRegressionNewton(max_iter=10)
    l1.fit(x_a, y_a)
    y_pre = l1.predict(x_a)
    accuracy = np.mean(y_pre == y_a)
    print(f"牛顿法训练出的权重:\n{l1.w}")
    print(f"牛顿法仅用 {l1.max_iter} 次迭代的准确率: {accuracy * 100:.2f}%")