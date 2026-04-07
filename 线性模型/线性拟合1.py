import numpy as np
np.random.seed(42)
def produce_dataset():
    m=200
    X1=np.random.randn(m,1)*10+100
    X2=np.random.randn(m,1)*1+3
    X3=X1 * 10.76 + np.random.randn(m, 1) * 0.01#X3和X1是共线性关系
    X_reg=np.hstack((X1,X2,X3))
    W=np.array([[5.0], [2.0], [0.5]])
    true_b = 10.0
    nosie=np.random.randn(m,1)*2.0
    Y_reg=X_reg.dot(W)+true_b+nosie
    return X_reg,Y_reg
class LinearRegressionRaw:
    def __init__(self,I=1e-4):
        self.I=I
        self.w = None
    def fit(self,X,Y):
        self.X=X
        self.Y=Y
        m, n = X.shape
        i1=np.ones((m,1))
        x_argue=np.hstack((i1,X))
        i2=np.eye(n+1)
        i2[0,0]=0#这里是为了不给b加上惩罚项
        try:
            self.w=np.linalg.inv((x_argue.T@x_argue+self.I*i2))@x_argue.T@self.Y
        except np.linalg.LinAlgError:
            self.w=np.linalg.pinv((x_argue.T@x_argue+self.I*i2))@x_argue.T@self.Y
    def predict(self,X):
        if self.w is None :#这里用的是is 去比较而不是用==，考试的时候一定要注意
            print("请先调用fit函数")
            return
        self.X=X
        m=self.X.shape[0]
        i1=np.ones((m,1))
        x_pre=np.hstack((i1,X))
        y_pre=x_pre@self.w
        return y_pre
if __name__=='__main__':
    X_reg,Y_reg=produce_dataset()
    l1=LinearRegressionRaw(0.01)
    l1.fit(X_reg,Y_reg)
    y_pre=l1.predict(X_reg)
print("模型求解出的增广权重向量：\n", l1.w)
