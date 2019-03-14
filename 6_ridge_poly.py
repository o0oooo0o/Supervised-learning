# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 13:46:51 2019

@author: 321
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

#data
n = 50
x=np.linspace(0,1,n)
s = np.sin(2*np.pi*x)
s=s.reshape(n,1)
num =0.3*np.random.randn(n,1)
y =s+num

alpha = 0.1
d=9
X=np.vander(x,d)

#ridge(polynomial)
ridge = Ridge(alpha=alpha).fit(X, y)

x_new =np.linspace(0,1,n)
X_new = np.vander(x_new,d)

predict = ridge.predict(X_new)

C = (np.dot(X.T, X)+(alpha*np.identity(X.shape[1])))
W = np.dot(np.dot(np.linalg.inv(C),X.T), y)
b = np.mean(y)-np.dot(W.T, np.mean(X, axis=0))
y_new = np.dot(X_new, W) + b

#plot
plt.subplots(1, 2, figsize=(12,4))
plt.subplot(121)
plt.title("sklearn Ridge (lamb={0}, d={1})".format(alpha, d))
plt.scatter(x,y)
plt.plot(x_new, predict, 'r-', label="predict")
plt.plot(x,s, 'k-', label="original")
plt.legend()

plt.subplot(122)
plt.title("numpy Ridge (lamb={0}, d={1})".format(alpha, d))
plt.scatter(x,y)
plt.plot(x_new, y_new, 'r-', label="predict")
plt.plot(x,s, 'k-', label="original")
plt.legend()
plt.show()

