import pandas as pd
import numpy as np
import pprint


# ==================== 前置辅助函数 (保持不变) ====================
def load_missing_data():
    dataset = [
        ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '是'],
        ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', np.nan, '是'],
        ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', '是'],
        ['青绿', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', '是'],
        [np.nan, '稍蜷', '浊响', '稍糊', '稍凹', '硬滑', '是'],
        ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', '否'],
        ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', '否'],
        ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '否'],
        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', '否'],
        [np.nan, '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', '否']
    ]
    columns = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感', '好瓜']
    df = pd.DataFrame(dataset, columns=columns)
    df['weight'] = 1.0  # 注入初始物理权重
    return df


def get_weighted_majority(df, label_col):
    return df.groupby(label_col)['weight'].sum().idxmax()


# ==================== Gini 数学核 (彻底替换信息熵) ====================
def calc_weighted_gini(df, label_col):
    """计算当前物理空间的加权 Gini 杂质度"""
    total_weight = df['weight'].sum()
    if total_weight <= 0:
        return 0.0

    gini = 1.0
    for val, sub_df in df.groupby(label_col):
        p = sub_df['weight'].sum() / total_weight
        gini -= p * p  # 核心物理突变：从 -p*log(p) 变为 1 - sum(p^2)

    return gini


def calc_gini_gain_with_missing(df, feature, label_col):
    """计算带有残缺样本惩罚的 Gini 增益"""
    total_weight = df['weight'].sum()

    # 相空间隔离
    df_valid = df[df[feature].notna()]
    df_valid_weight = df_valid['weight'].sum()

    # 物理死锁：当前节点在当前特征上全军覆没 (全为 NaN)
    if df_valid_weight <= 0:
        return -1.0

    rou = df_valid_weight / total_weight

    # 计算无缺失子集的纯净 Gini 基准值
    gini_base = calc_weighted_gini(df_valid, label_col)

    # 计算按特征切分后的条件 Gini 杂质度
    feat_gini = 0.0
    for val, sub_df in df_valid.groupby(feature):
        # 注意：分母必须是纯净子集的绝对质量 (df_valid_weight)
        r = sub_df['weight'].sum() / df_valid_weight
        feat_gini += r * calc_weighted_gini(sub_df, label_col)

    # Gini 增益 = (划分前的杂质度 - 划分后的杂质度) * 衰减系数
    return rou * (gini_base - feat_gini)


# ==================== 极简递归引擎 ====================
def build_gini_tree_minimal(df, label_col, features):
    # 1. 绝对纯度达标
    if len(df[label_col].unique()) == 1:
        return df[label_col].iloc[0]

    # 2. 特征耗尽 或 样本特征完全重叠死锁
    if len(features) == 0 or len(df[features].drop_duplicates()) == 1:
        return get_weighted_majority(df, label_col)

    # 3. 寻优机制 (使用 Gini Gain)
    best_fea = None
    max_gain = -1.0
    for fea in features:
        gain = calc_gini_gain_with_missing(df, fea, label_col)
        if gain > max_gain:
            best_fea = fea
            max_gain = gain

    if best_fea is None:
        return get_weighted_majority(df, label_col)

    # 4. 初始化子树字典挂载点
    tree = {best_fea: {}}

    # 5. 核心：提取循环外部执行物理样本的撕裂，压榨极限性能
    df_valid = df[df[best_fea].notna()]
    df_missing = df[df[best_fea].isna()]
    sub_feature = [f for f in features if f != best_fea]

    # 仅遍历当前特征存在的有效物理态
    for v in df_valid[best_fea].unique():
        sub_df_valid = df_valid[df_valid[best_fea] == v].copy()
        sub_weight = sub_df_valid['weight'].sum()
        r1 = sub_weight / df_valid['weight'].sum()

        # 衰减叠加态样本 (缺失值) 的质量
        sub_df_missing = df_missing.copy()
        if not sub_df_missing.empty:
            sub_df_missing['weight'] = r1 * sub_df_missing['weight']

        # 缝合后，直接利用字典隐式挂载语法进行下一层递归
        sub_df = pd.concat([sub_df_valid, sub_df_missing], ignore_index=True)
        tree[best_fea][v] = build_gini_tree_minimal(sub_df, label_col, sub_feature)

    return tree


# ==================== 执行系统 ====================
if __name__ == "__main__":
    pd.options.mode.chained_assignment = None

    dataset = load_missing_data()
    features = list(dataset.columns[:-2])  # 剔除 '好瓜' 和 'weight'

    tree = build_gini_tree_minimal(dataset, "好瓜", features)

    print("基于 Gini 增益的极简递归决策树结构:\n")
    pprint.pprint(tree, sort_dicts=False, width=40)