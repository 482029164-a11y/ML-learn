import numpy as np
import pandas as pd


def load_and_vectorize_data():
    dataset = [
        ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.697, 0.460, '是'],
        ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', 0.774, 0.376, '是'],
        ['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', 0.556, 0.215, '是'],
        ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', 0.403, 0.237, '否'],
        ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', 0.481, 0.149, '否'],
        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘', 0.666, 0.091, '否']
    ]
    columns = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感', '密度', '含糖率', '好瓜']
    df = pd.DataFrame(dataset, columns=columns)
    df['好瓜'] = df['好瓜'].apply(lambda x: 1 if x == '是' else 0)
    discrete_features = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感']
    df_encoded = pd.get_dummies(df, columns=discrete_features, dtype=float)
    return df_encoded


def sigmoid(z):
    z = np.clip(z, -250, 250)
    return 1 / (1 + np.exp(-z))


def fit(x, y, learn_rate=0.01, epoches=100):
    m, n = x.shape
    w = np.zeros(n + 1)
    i = np.ones((m, 1))
    x = np.hstack((i, x))

    # 提前剥离标签的 Pandas 索引，化为纯 NumPy 数组 (m,)
    y_vals = y.values

    for _ in range(epoches):
        p = x @ w
        p = sigmoid(p)
        # 绝对干净的元素级减法
        dz = p - y_vals
        grade = x.T @ dz
        ave_grade = grade / m
        w -= learn_rate * ave_grade
    return w


class treenode:
    def __init__(self):
        self.left = None
        self.right = None
        self.lable = None
        self.w = None


def buildtree(x, y):
    m, n = x.shape
    node = treenode()
    if len(y.unique()) == 1:
        node.lable = y.iloc[0]
        return node
    if len(y) <= 2:
        node.lable = y.value_counts().idxmax()
        return node

    w = fit(x, y)
    i1 = np.ones((m, 1))
    x1 = np.hstack((i1, x))
    z = x1 @ w
    p = sigmoid(z)

    left_mask = p >= 0.5
    right_mask = p < 0.5

    if sum(left_mask) == 0 or sum(right_mask) == 0:
        # 修复拼写错误：idxmax()
        node.lable = y.value_counts().idxmax()
        return node

    node.w = w
    node.left = buildtree(x[left_mask], y[left_mask])
    node.right = buildtree(x[right_mask], y[right_mask])
    return node


if __name__ == "__main__":
    df = load_and_vectorize_data()
    y = df["好瓜"]
    x = df.drop("好瓜", axis=1)
    node = buildtree(x, y)

    # 打印测试：你应该能看到一个包含 19 个浮点数的一维数组
    np.set_printoptions(precision=4, suppress=True)  # 让输出更整洁
    print("根节点超平面权重 (包含 w0 偏置):\n", node.w)