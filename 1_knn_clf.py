# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 17:00:00 2019

@author: 321
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

#data
data = np.genfromtxt('banana.csv', delimiter=",")
train,test = train_test_split(data, test_size=0.3, random_state=0)
x_train = train[:,0:2]
x_test = test[:,0:2]
y_train = train[:,2]
y_test = test[:,2]

#sklearn training
k = 25
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(x_train, y_train)

n = 50
X1, X2 = np.meshgrid(np.linspace(-3.5, 3, n), np.linspace(-3, 3.5, n))
XX = np.hstack((np.reshape(X1, (n*n,1)), np.reshape(X2, (n*n,1))))

predict = knn.predict(XX)
predict = np.reshape(predict, (n,n))

#numpy training
result = np.zeros((n,n))
for i in range(0,n):
    for j in range(0,n):
        dists = []
        for z in range(0, train.shape[0]):
            dist = np.sqrt((x_train[z,0]-X1[i,j])**2 + (x_train[z,1]-X2[i,j])**2)
            dists.append(dist)
            
        idx = np.argsort(dists)
        neighborIdx = idx[:k]
        
        if np.sum(y_train[neighborIdx]) > 0:
            result[i,j] = 1
        elif np.sum(y_train[neighborIdx]) == 0:
            result[i,j] = y_train[neighborIdx[0]]
        else:
            result[i,j] = -1
            
#plot
plt.subplots(1,2, figsize=(12,4))
plt.subplot(1,2,1)
mm=x_test[y_test==-1]
pp=x_test[y_test==1]
plt.title("scikit predict (k={})".format(k))
plt.scatter(mm[:,0],mm[:,1], marker='x')
plt.scatter(pp[:,0],pp[:,1], marker='.')
plt.contour(X1, X2, predict, 0, colors='k', linewidth=.1)

plt.subplot(1,2,2)
plt.title("numpy result (k={})".format(k))
plt.scatter(mm[:,0],mm[:,1], marker='x')
plt.scatter(pp[:,0],pp[:,1], marker='.')
plt.contour(X1, X2, result, 0, colors='k', linewidth=.1)
plt.show()

