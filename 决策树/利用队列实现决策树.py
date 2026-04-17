import pandas as pd
import numpy as np
from collections import deque
import pprint
from math import log2

# ==================== 1. 数据集构建 ====================
def load_discrete_data():
    """
    构建纯离散相空间的西瓜数据集 (剥离了连续特征，专注于树的拓扑结构生成)
    """
    dataset = [
        ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '是'],
        ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '是'],
        ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', '是'],
        ['青绿', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', '是'],
        ['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '硬滑', '是'],
        ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', '否'],
        ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', '否'],
        ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '否'],
        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', '否'],
        ['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', '否']
    ]
    columns = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感', '好瓜']
    return pd.DataFrame(dataset, columns=columns)
def cal_entroph(df,lable):
    en=0
    for val,sub_df in df.groupby(lable):
        p=len(sub_df)/len(df)
        en=-p*log2(p)
    return en
def cal_gain(df,feature,lable):
    ent=cal_entroph(df,lable)
    max_gain=-1
    best_fea=None
    for v in feature:
        gain = 0
        for val,sub_df in df.groupby(v):
            p=len(sub_df)/len(df)
            gain+=p*cal_entroph(sub_df,lable)
        gain1=ent-gain
        if gain1>max_gain:
            max_gain=gain1
            best_fea=v
    return best_fea
def getmax(df,lable):
    return df[lable].value_counts().idxmax()
def buildtrees(df,feature,lable,max_depth=5):
    queue=deque()
    tree_anchor={}
    s=(df,feature,tree_anchor,"root",0)
    queue.append(s)
    while(len(queue)>0):
        curr_df, curr_feats, parent_ptr, branch_key, curr_depth=queue.popleft()#因为是双端队列，所以区分左右
        if(len(curr_df[lable].unique())==1):
            parent_ptr[branch_key] = curr_df[lable].iloc[0]
            continue
        if len(curr_feats)==0 or len(curr_df[curr_feats].drop_duplicates())==1:
            parent_ptr[branch_key] = getmax(curr_df, lable)
            continue
        if curr_depth >= max_depth:
            parent_ptr[branch_key] = getmax(curr_df, lable)
            continue
        bestfea=cal_gain(df, feature, lable)
        current_node_dict = {bestfea: {}}
        parent_ptr[branch_key]=current_node_dict
        sub_fea=[f for f in feature if f!=bestfea]
        for val,sub_df in curr_df.groupby(bestfea):
            queue.append((sub_df,sub_fea,current_node_dict[bestfea],val,curr_depth+1))
    return tree_anchor["root"]
if __name__ == "__main__":
    # 加载数据
    df = load_discrete_data()
    all_features = df.columns[:-1]  # 剥离最后一列标签
    target_label = '好瓜'

    print("启动底层 BFS 树生成引擎 (设定最大物理深度 = 2)...\n")

    # 执行生成
    decision_tree_dict = buildtrees(df, all_features, target_label, max_depth=2)

    # 使用 pprint 以高可读性的嵌套结构打印内存中的树字典
    print("决策树内存映射结构:")
    pprint.pprint(decision_tree_dict)
