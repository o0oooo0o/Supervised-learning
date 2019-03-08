# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 19:31:39 2019

@author: 321
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import mglearn

#data
x,y = mglearn.datasets.make_forge()
y=np.reshape(y,(-1,1))
logistic = LogisticRegression().fit(x,y)

#training
n=1000
x1 = np.linspace(np.min(x[:,0]), np.max(x[:,0]), n)
x2 = np.linspace(np.min(x[:,1]), np.max(x[:,0]), n)
X1, X2 = np.meshgrid(x1, x2)
XX = np.hstack((np.reshape(X1, (n*n,1)), np.reshape(X2, (n*n,1))))
#grid = np.c_[X1.ravel(), X2.ravel()]  #XX = grid

predict = logistic.predict(XX)
predict = np.reshape(predict, (n,n))
#probs = logistic.predict_proba(XX)[:, 1].reshape(X1.shape)


alpha = 30
def sigmoid(z):
    return 1 / (1+np.exp(-z))

def logistic_loss(y,H,W):
    return (-np.mean(y*np.log(H) + (1-y)*np.log(1-H)) + (alpha*np.mean(W*W)))

W = np.zeros((2,1))
b = np.zeros((1,1))
learning_rate = 0.01

m = len(y)

for epoch in range(1000):
    z = np.matmul(x,W)+b
    H = sigmoid(z)
    loss = logistic_loss(y,H,W)
    dz = H-y
    if epoch ==0:
        dw = (1/m)*np.matmul(x.T, dz)
        db = np.sum(dz)
    else:
        dw = (1/m)*np.matmul(x.T, dz) + (alpha/m * dw)
        db = np.sum(dz)+(alpha/m * db)
    
    W = W - learning_rate*dw
    b = b - learning_rate*db
    
#    if epoch % 1000 == 0:
#        print(loss)
        
result = []
ZZ = np.matmul(XX,W)+b
for i in sigmoid(ZZ):
    if i > 0.5:
        result.append(1)
    else:
        result.append(0)
        
result = np.reshape(result, (n,n))

#plot
plt.subplots(1,2, figsize=(12,4))
plt.subplot(1,2,1)
plt.title("sklearn Logistic (learning rate={})".format(learning_rate))
mm=x[y[:,0]==0]
pp=x[y[:,0]==1]
plt.scatter(mm[:,0],mm[:,1], marker='x')
plt.scatter(pp[:,0],pp[:,1], marker='.')
plt.ylim(np.min(x[:,1])*1.2, np.max(x[:,1])*1.1)
plt.contour(X1, X2, predict, 0, colors='k', linewidth=.1)

plt.subplot(1,2,2)
plt.title("Logistic (learning rate={})".format(learning_rate))
plt.scatter(mm[:,0],mm[:,1], marker='x')
plt.scatter(pp[:,0],pp[:,1], marker='.')
plt.ylim(np.min(x[:,1])*1.2, np.max(x[:,1])*1.1)
plt.contour(X1, X2, result, 0, colors='k', linewidth=.1)
plt.show()