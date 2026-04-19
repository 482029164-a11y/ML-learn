import numpy as np


# 物理底层的激活函数与导数
def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))


def sigmoid_derivative(output):
    return output*(1-output)


class BpNetwork:
    def __init__(self,input_dim,hidden_dim,output_dim,learning_rate=0.01,epochs=1000):
        np.random.seed(42)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.W1=np.random.randn(input_dim,hidden_dim)*0.1
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.1
        self.b2 = np.zeros((1, output_dim))
    def train(self, X, Y, learning_rate, epochs, mode='standard'):
        n_samples = X.shape[0]
        loss_history=[]
        count=0
        for i in range(epochs):
            if mode=='standard':
                loss = 0
                for _ in range(n_samples):
                    x=X[_:_+1]
                    y=Y[_:_+1]
                    hidden=sigmoid(x@self.W1+self.b1)#1*hidden
                    out=sigmoid(hidden@self.W2+self.b2)#1*out
                    error=out-y
                    loss += np.sum(error ** 2) / 2.0
                    g_=error*sigmoid_derivative(out)#哈达玛内积,维数为：1*out
                    en=sigmoid_derivative(hidden)*np.dot(g_,self.W2.T)
                    self.W2-=learning_rate*np.dot(hidden.T,g_)
                    self.b2-=learning_rate*g_
                    self.W1-=learning_rate*np.dot(x.T,en)
                    self.b1-=learning_rate*en
                    count+=1
                loss_history.append(loss/n_samples)
            elif mode=='accumulated':
                loss=0
                hidden = sigmoid(X @ self.W1 + self.b1)
                out = sigmoid(hidden @ self.W2 + self.b2)  # n_samples*out
                error = out - Y# n_samples*out
                loss=np.sum(error ** 2) / (2.0*n_samples)
                g_ = error * sigmoid_derivative(out)  # 哈达玛内积,维数为：n*out
                en = sigmoid_derivative(hidden) * np.dot(g_, self.W2.T)# 哈达玛内积,维数为：n*hidden
                self.W2 -= learning_rate * np.dot(hidden.T,g_)/ n_samples
                self.W1 -= learning_rate * np.dot(X.T,en)/ n_samples
                self.b2 -= learning_rate * np.sum(g_, axis=0, keepdims=True) / n_samples
                self.b1 -= learning_rate * np.sum(en, axis=0, keepdims=True) / n_samples
                count += 1
                loss_history.append(loss)
        return loss_history,count
if __name__ == "__main__":
    X_data = np.array([
        [0.697, 0.460], [0.774, 0.376], [0.634, 0.264], [0.608, 0.318],
        [0.556, 0.215], [0.403, 0.237], [0.481, 0.149], [0.437, 0.211],
        [0.666, 0.091], [0.243, 0.267], [0.245, 0.057], [0.343, 0.099],
        [0.639, 0.161], [0.657, 0.198], [0.360, 0.370], [0.593, 0.042],
        [0.719, 0.103]
    ])

    # 绝对物理拓扑突变：将一维标签转化为 17x2 的独热编码矩阵 (One-Hot)
    # [1, 0] 代表反例(坏瓜)，[0, 1] 代表正例(好瓜)
    y_raw = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    y_data = np.zeros((17, 2))
    for i, label in enumerate(y_raw):
        y_data[i, label] = 1

    # 实例化网络：2个输入特征，10个隐藏节点，2个输出节点
    net_std = BpNetwork(input_dim=2, hidden_dim=10, output_dim=2)
    net_acc = BpNetwork(input_dim=2, hidden_dim=10, output_dim=2)

    epochs = 1000
    lr = 0.5

    std_loss, std_updates = net_std.train(X_data, y_data, lr, epochs, mode='standard')
    acc_loss, acc_updates = net_acc.train(X_data, y_data, lr, epochs, mode='accumulated')

    print(f"=== 物理演化终局评估 (双输出节点架构) ===")
    print(f"[标准 BP] 权重写操作次数: {std_updates}, 最终残余误差: {std_loss[-1]:.4f}")
    print(f"[累积 BP] 权重写操作次数: {acc_updates}, 最终残余误差: {acc_loss[-1]:.4f}")