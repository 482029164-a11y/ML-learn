# 在最优特征的每个取值上，执行真实的物理切割并向下递归
import pandas as pd
# ==================== 1. 物理环境与数据加载 ====================
def load_pruning_data():
    # 西瓜书表 4.2：训练集
    train_data = [
        ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '是'],
        ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '是'],
        ['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '是'],
        ['青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '是'],
        ['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', '是'],
        ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', '否'],
        ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', '否'],
        ['乌黑', '稍蜷', '浊响', '清晰', '平坦', '软粘', '否'],
        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘', '否'],
        ['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', '否']
    ]
    # 西瓜书表 4.2：验证集
    val_data = [
        ['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '是'],
        ['浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '是'],
        ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', '是'],
        ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑', '否'],
        ['浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑', '否'],
        ['青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', '否'],
        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', '否']
    ]
    columns = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感', '好瓜']
    return pd.DataFrame(train_data, columns=columns), pd.DataFrame(val_data, columns=columns)
def cal_geni(df,lable_clo):
    lable=df[lable_clo]
    gini=1
    for counts in lable.value_counts():
        p=counts/len(lable)
        gini-=p*p
    return gini
def cal_gini_index(df,feature,lable_clo):
    gini_index=0
    for value,counts in df.groupby(feature):
        p=len(counts)/len(df)
        gini_index+=p*cal_geni(counts,lable_clo)
    return gini_index
def getmax(df,lable_clo):
    if df.empty:
        return None
    return df[lable_clo].value_counts().idxmax()
def buildtree(df,df1,feature,lable_clo):#df 为训练集，df1为验证集
    #两个递归结束的条件
    if (len(df[lable_clo].unique())==1):
        return df[lable_clo].iloc[0]
    if (len(feature)==0)or len(df[feature].drop_duplicates())==1:
        return getmax(df,lable_clo)
    #根据基尼指数寻找最优特征
    min_gini = float('inf')
    best_feature=" "
    for features in feature:
        temp=cal_gini_index(df, features, lable_clo)
        if temp<min_gini:
            min_gini=temp
            best_feature=features
    #计算不划分时的正确个数
    maxclass=getmax(df,lable_clo)
    base=0
    base+=sum(df1[lable_clo]==maxclass)#这里是用了广播机制
    #计算划分之后的正确个数
    cut=0
    for value,sub_df in df.groupby(best_feature):
        sub_df1=df1[df1[best_feature]==value]
        if not sub_df.empty:
            sub_maxclass=getmax(sub_df,lable_clo)
            cut+=sum(sub_df1[lable_clo]==sub_maxclass)
    if base>=cut:
        return maxclass#如果不划分更好就停止
    tree={best_feature:{}}#如果划分更好，那么就创建一颗树
    for counts, sub_df in df.groupby(best_feature):
        sub_feature=[f for f in feature if f!=best_feature]#记录剩下的特征集合
        sub_df1=df1[df1[best_feature]==counts]#需要递归的都需要算sub
        tree[best_feature][counts]=buildtree(sub_df,df1,sub_feature,lable_clo)#递归调用树函数
    return tree
if __name__=="__main__":
    train,val=load_pruning_data()
    features=train.columns[:-1]
    trees=buildtree(train,val,features,"好瓜")
    print(trees)


