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
def cal_gini(df,lable_clo):
    lable=df[lable_clo]
    gini=1
    for counts in lable.value_counts():
        p=counts/len(lable)
        gini-=p*p
    return gini
def cal_gini_dx(df,feature,lable_clo):
    gini_index=0
    for v,sub_df in df.groupby(feature):
        p=len(sub_df)/len(df)
        gini_index+=p*cal_gini(sub_df,lable_clo)
    return gini_index
def getmax(df,lable_clo):
    if df.empty:
        return None
    else:
        return df[lable_clo].value_counts().idxmax()
def buildtree(train,feature,lable_clo):
    if len(train[lable_clo].unique())==1:
        return train[lable_clo].iloc[0]
    if len(feature)==0 or len(train[feature].drop_duplicates())==1:
        return getmax(train, lable_clo)
    min_gini=float("inf")
    best_feature=""
    for fea in feature:
        temp=cal_gini_dx(train,fea,lable_clo)
        if temp<min_gini:
            min_gini=temp
            best_feature=fea
    tree={best_feature:{}}
    for value,sub_df in train.groupby(best_feature):
        sub_features=[f for f in feature if f!=best_feature]
        tree[best_feature][value]=buildtree(sub_df,sub_features,lable_clo)
    return tree
#按行去预测
def pridict_row(tree,row):
    if not isinstance(tree, dict): return tree  # isinstance(数据, 类型)，判断数据等不等于该类型，就是看是否到叶子节点
    feature =list(tree.keys())[0]#获取当前树的根结点
    val=row[feature]#获取当前特征的特征值
    if val in tree[feature]:
        return pridict_row(tree[feature][val],row)
    else:
        return None
def post_prune(tree,train,val,lable_clo):
    if not isinstance(tree, dict): return tree #递归程序的出口
    feature = list(tree.keys())[0]  # 获取当前树的根结点
    for v in tree[feature].keys():
        sub_train=train[train[feature]==v]
        sub_val=val[val[feature]==v]
        tree[feature][v]=post_prune(tree[feature][v],sub_train,sub_val,lable_clo)#递归的时候要默认程序可以完成该操作
    #下面开始正式进行对这一层进行剪枝处理
    maxclass=getmax(train,lable_clo)
    cut=sum(val[lable_clo]==maxclass)#计算不分枝的基准正确个数
    base=sum(val.apply(lambda row:pridict_row(tree,row)==row[lable_clo],axis=1))
    if(cut>=base):
        return maxclass
    else:
        return tree
if __name__=="__main__":
    train,val=load_pruning_data()
    feature=train.columns[:-1]
    trees=buildtree(train, feature, "好瓜")
    print(trees)
    post_tree=post_prune(trees, train, val, "好瓜")
    print(post_tree)



