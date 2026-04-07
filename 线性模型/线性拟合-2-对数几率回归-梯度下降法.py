# 本例没有做硬性预测，硬性预测在后面的牛顿法会体现
import numpy as np
def pro_data():
    m=150
    m2=300
    x1=np.random.randn(m,2)+np.array([2,2])
    x2=np.random.randn(m,2)+np.array([-2,-2])
    y1=np.zeros((m,1))
    y2=np.ones((m,1))
    x_clf=np.vstack((x1,x2))
    y_clf=np.vstack((y1,y2))
    # 下面三段代码是在对已有的数据进行洗牌处理
    permutation=np.random.permutation(m2)
    x_clf=x_clf[permutation]
    y_clf=y_clf[permutation]
    return x_clf,y_clf
class LogisticRegressionGD:
    def __init__(self,l=0.01,e=1000):
        self.l=l
        self.e=e
        self.w=None
    def sigmoid(self,z):
        z_safe=np.clip(z,-250,250)
        return 1.0/(1.0+np.exp(-z_safe))
    def fit(self,x,y):
        self.x=x
        self.y=y
        m,n=x.shape
        i1=np.ones((m,1))
        x1=np.hstack((i1,self.x))
        self.w=np.ones((n+1,1))
        for e in range(self.e):
            z=x1@self.w
            p=self.sigmoid(z)
            erro=p-y
            g=x1.T@erro
            g_ave=g/m#这里计算了平均梯度
            self.w=self.w-self.l*g_ave
    def predict(self,x):
        self.x=x
        if self.w is None:
            print("请先调用fit函数")
            return
        m=x.shape[0]
        i1=np.ones((m,1))
        x1=np.hstack((i1,self.x))
        y_pre=x1@self.w
        return y_pre
if __name__=="__main__":
    x,y=pro_data()
    l1=LogisticRegressionGD()
    l1.fit(x,y)
    l1.predict(x)
