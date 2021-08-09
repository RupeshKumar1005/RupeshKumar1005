import numpy as np
class Linear_Regression:
    def hypo(self,X,theta):
        return np.dot(X,theta)
    def gradient(self,X,y,theta):
        h=self.hypo(X,theta)
        grad=np.dot(np.transpose(X),h-y)
        return grad
    def cost(self,X,y,theta):
        h=self.hypo(X,theta)
        J=np.dot(np.transpose(h-y),h-y)
        J/=2
        return J[0][0]
    def gradient_descent(self,X,y,learn):
        theta=np.zeros((X.shape[1],1))
        convergence=0.00000005
        cost,prev_cost=(100,200)
        while(abs(prev_cost-cost)>convergence):
            prev_cost=cost
            theta=theta-(learn*(self.gradient(X,y,theta)))
            cost=self.cost(X,y,theta)
        return theta
    def normal_eq(self,X,y):
        theta=np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X),X)),np.transpose(X)),y)
        return theta
x=[[1,2],[1,5],[1,7],[1,9],[1,11],[1,13],[1,15]]
y=[[23],[45],[98],[123],[140],[178],[200]]
x=np.array(x)
y=np.array(y)
l=Linear_Regression()
thet=l.gradient_descent(x,y,0.001)  
print(thet)
thet2=l.normal_eq(x,y)
print(thet)
