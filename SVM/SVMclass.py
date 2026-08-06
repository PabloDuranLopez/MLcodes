import numpy as np
from scipy.optimize import minimize 

class SVM:
    def __init__(self, kernel='lineal', C=None, gamma=1):
        #C=None= Hard Margin C!=None Soft Margin
        self.kernel= kernel
        self.C=C
        self.gamma=gamma

    def kernel_function(self, x1, x2):
        if self.kernel=='lineal':
            return x1@x2.T
        elif self.kernel=='RBF':
            diff=x1[:,None]-x2[None,:]
            dist=np.sum(diff**2,axis=2)
            return np.exp(-self.gamma*dist)
        else:
            raise ValueError("Kernel no soportado.")
        

    def fit(self,x,y):
        alpha=np.random.rand(len(y))
        k=self.kernel_function(x,x)
        z=np.outer(y,y)*k
        def minimize_function(alphas):
            return 0.5*alphas@z@alphas - np.sum(alphas)
        cons=({"type":"eq","fun": lambda alpha: np.sum(alpha*y)})
        bounds=[]
        if self.C is not None:
            bounds=[(0,self.C)]*len(y)
            opt=minimize(minimize_function, alpha, constraints=cons, bounds=bounds, method="SLSQP")
            a=opt.x
            sv=(a>1e-5) & (a<self.C - (1e-5))
        else:
            bounds=[(0,None)]*len(y)
            opt=minimize(minimize_function, alpha, constraints=cons, bounds=bounds, method="SLSQP")
            a=opt.x
            sv=(a>1e-5)
        b=np.mean(y[sv]-k[sv]@(y*a))
        self.sv=sv
        self.b=b
        self.x=x
        self.a=a
        self.y=y
        return self

    def predict(self,X):
        k=self.kernel_function(X,self.x)
        y_pred=k@(self.a*self.y) + self.b
        return np.sign(y_pred)



        
        
    


        
    
        



