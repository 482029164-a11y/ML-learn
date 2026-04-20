import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(output):
    return output * (1 - output)


class BpNetwork:
    def __init__(self, input_dim, hidden_dim, output_dim):
        np.random.seed(42)
        self.W1=np.random.rand(input_dim,hidden_dim)*0.1
        self.b1=np.zeros((1,hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.1
        self.b2 = np.zeros((1, output_dim))
    def train(self, X, Y, initial_lr, epochs, mode='adaptive'):
        n_samples = X.shape[0]
        loss_history = []
        learning_rate = initial_lr

        # 为了启动判定，先计算第 0 时刻的绝对物理初始误差
        hidden=np.dot(X,self.W1)+self.b1
        out = np.dot(hidden, self.W2) + self.b2
        current_loss=np.sum(out-Y)/(2*n_samples)
        for i in range(epochs):
            if mode == 'adaptive':
                # 1. 在当前状态下，计算出标准梯度
                g_ = (out - Y) * sigmoid_derivative(out)
                en = sigmoid_derivative(hidden) * np.dot(g_, self.W2.T)
                dW2 = np.dot(hidden.T, g_) / n_samples
                db2 = np.sum(g_, axis=0, keepdims=True) / n_samples
                dW1 = np.dot(X.T, en) / n_samples
                db1 = np.sum(en, axis=0, keepdims=True) / n_samples

                # 2. 试探性步进（产生平行时空的分支，不污染当前真实权重）
                W1_new = self.W1 - learning_rate * dW1
                b1_new = self.b1 - learning_rate * db1
                W2_new = self.W2 - learning_rate * dW2
                b2_new = self.b2 - learning_rate * db2

                # 3. 评估平行时空的误差
                hidden_new = sigmoid(np.dot(X, W1_new) + b1_new)
                out_new = sigmoid(np.dot(hidden_new, W2_new) + b2_new)
                new_loss = np.sum((out_new - Y) ** 2) / (2.0 * n_samples)

                # 4. 绝对物理判决：对撞结果
                if new_loss < current_loss:
                    # 判决：方向正确。全盘接收新状态，并踩油门加速 (1.05倍)
                    self.W1, self.b1 = W1_new, b1_new
                    self.W2, self.b2 = W2_new, b2_new
                    hidden, out = hidden_new, out_new
                    current_loss = new_loss

                    learning_rate *= 1.05
                    loss_history.append(current_loss)
                else:
                    # 判决：越界砸墙。拒绝新状态，保持原权重不动，并急刹车降速 (0.5倍)
                    # 注意：此时底层的 self.W1 等变量根本没被修改，完美实现了物理回滚
                    learning_rate *= 0.5
                    loss_history.append(current_loss)

        return loss_history, learning_rate


if __name__ == "__main__":
    X_data = np.array([
        [0.697, 0.460], [0.774, 0.376], [0.634, 0.264], [0.608, 0.318],
        [0.556, 0.215], [0.403, 0.237], [0.481, 0.149], [0.437, 0.211],
        [0.666, 0.091], [0.243, 0.267], [0.245, 0.057], [0.343, 0.099],
        [0.639, 0.161], [0.657, 0.198], [0.360, 0.370], [0.593, 0.042],
        [0.719, 0.103]
    ])

    y_raw = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    y_data = np.zeros((17, 2))
    for i, label in enumerate(y_raw):
        y_data[i, label] = 1

    net_adaptive = BpNetwork(input_dim=2, hidden_dim=10, output_dim=2)

    epochs = 1000
    initial_lr = 0.5

    ada_loss, final_lr = net_adaptive.train(X_data, y_data, initial_lr, epochs, mode='adaptive')

    print(f"=== 物理演化终局评估 (自适应学习率) ===")
    print(f"初始学习率设定: {initial_lr}")
    print(f"最终收敛学习率: {final_lr:.4f} (引擎自动调节结果)")
    print(f"最终残余误差:   {ada_loss[-1]:.6f}")