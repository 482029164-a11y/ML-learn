import numpy as np
import pandas as pd
from math import log2

# 机考标准数据集下发：西瓜数据集 2.0
def load_watermelon_data():
    dataset = [
        ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '是'],
        ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '是'],
        ['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '是'],
        ['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '是'],
        ['浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '是'],
        ['青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '是'],
        ['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', '是'],
        ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', '是'],
        ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑', '否'],
        ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', '否'],
        ['浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑', '否'],
        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘', '否'],
        ['青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', '否'],
        ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', '否'],
        ['乌黑', '稍蜷', '浊响', '清晰', '平坦', '软粘', '否'],
        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', '否'],
        ['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', '否']
    ]
    columns = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感', '好瓜']
    return pd.DataFrame(dataset, columns=columns)#一行一行扫描数组，并且将columns自动变成特征标签
#计算信息熵
def cal_entroph(df,label):
    e=0
    dm=df[label]
    for counts in dm.value_counts():
        p=counts/len(dm)
        e+=-p*log2(p)
    return e
# 计算增益
def calc_info_gain(df, feature_col, label_col):
    base=cal_entroph(df,label_col)
    dm=df.groupby(feature_col)
    new_entrophy=0#value是df的子表
    for index,value in dm:
        p=len(value)/len(df)
        new_entrophy+=p*cal_entroph(value,label_col)
    return base-new_entrophy #返回GAIN

def getmax(df,lable_clo):
    return df[lable_clo].value_counts().idxmax()#返回类别最多的下标

def buildtree(df,lable_clo,feature):
    lables=df[lable_clo]
    if (len(lables.unique())==1):
        return lables.iloc[0]
    if len(feature)==0 or len(df[feature].drop_duplicates())==1:
        return getmax(df,lable_clo)
    max=0
    t_f=None
    for fea in feature:
        temp=calc_info_gain(df, fea, lable_clo)
        if temp>=max:
            max=temp
            t_f=fea

    tree={t_f:{}}
    for v,sub_df in df.groupby(t_f):
        sub_f=[f1 for f1 in feature if f1!=t_f]
        tree[t_f][v]=buildtree(sub_df,lable_clo,sub_f)
    return tree
if __name__=="__main__":
    dataset=load_watermelon_data()
    features = list(dataset.columns[:-1])  # 剔除最后一列标签
    tree=buildtree(dataset, "好瓜", features)
    print(tree)