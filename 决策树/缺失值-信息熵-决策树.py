import pandas as pd
import numpy as np
import pprint
from math import log2


# ==================== 前置辅助函数 (保持物理运算原貌) ====================
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


def calc_weighted_entropy(df, label_col):
    total_weight=df['weight'].sum()
    if total_weight<=0:
        return 0.0
    en=0
    for val,sub_df in df.groupby(label_col):
        p=sub_df['weight'].sum()/total_weight
        if p>0: en-=p*log2(p)
    return en


def calc_gain_with_missing(df, feature, label_col):
    total_weight=df['weight'].sum()
    df_valid=df[df[feature].notna()]
    df_valid_weight=df_valid['weight'].sum()
    rou=df_valid_weight/total_weight
    if rou<=0: return -1
    en0=calc_weighted_entropy(df_valid, label_col)
    feat = 0
    for val,sub_df in df_valid.groupby(feature):
        r=sub_df['weight'].sum()/df_valid_weight
        feat+=r*calc_weighted_entropy(sub_df, label_col)
    return rou*(en0-feat)
# ==================== 极简递归引擎 (你的期望重构版) ====================
def build_c45_tree_minimal(df, label_col, features):
    # 1. 绝对纯度达标
    if len(df[label_col].unique())==1:
        return df[label_col].iloc[0]
    # 2. 特征耗尽 或 样本特征完全重叠死锁
    if len(features)==0 or len(df[features].drop_duplicates())==1:
        return get_weighted_majority(df, label_col)
    # 3. 寻优机制
    best_fea=None
    max_gain=-1
    for fea in features:
        gain=calc_gain_with_missing(df, fea, label_col)
        if gain>max_gain:
            best_fea=fea
            max_gain=gain
    if best_fea==None:
        return get_weighted_majority(df, label_col)
    # 4. 初始化子树字典挂载点
    tree={best_fea:{}}
    # 5. C4.5 核心：在循环开始前，极其利落地完成一次物理相空间的彻底撕裂
    df_valid = df[df[best_fea].notna()]
    df_missing = df[df[best_fea].isna()]  # 移到循环外部！
    sub_feature = [f for f in features if f != best_fea]

    # 仅遍历当前特征存在的有效物理态
    for v in df_valid[best_fea].unique():
        sub_df_valid = df_valid[df_valid[best_fea] == v].copy()
        sub_weight = sub_df_valid['weight'].sum()
        r1 = sub_weight / df_valid['weight'].sum()

        # B. 衰减叠加态样本，此时直接 copy 外部已经切片好的数据，极速执行
        sub_df_missing = df_missing.copy()
        if not sub_df_missing.empty:  # 加上防御性拦截，防止空矩阵运算
            sub_df_missing['weight'] = r1 * sub_df_missing['weight']
        # C. 缝合
        sub_df = pd.concat([sub_df_valid, sub_df_missing], ignore_index=True)
        tree[best_fea][v] = build_c45_tree_minimal(sub_df, label_col, sub_feature)
    return tree


# ==================== 执行系统 ====================
if __name__ == "__main__":
    pd.options.mode.chained_assignment = None

    dataset = load_missing_data()
    features = list(dataset.columns[:-2])  # 剔除 '好瓜' 和 'weight'

    tree = build_c45_tree_minimal(dataset, "好瓜", features)

    print("极简递归 C4.5 决策树结构:\n")
    pprint.pprint(tree, sort_dicts=False, width=40)