# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 17:14:16 2019

@author: 321
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

#data
n = 50
x=np.linspace(-3,3,n)
x=np.reshape(x, (-1,1))
s = 80*x - 100
s=s.reshape(n,1)
num =70*np.random.randn(n,1)
y =s+num
alpha = 0.01

#training
ridge = Ridge(alpha=alpha).fit(x, y)

x_new = np.linspace(-3,3,n).reshape(-1,1)

predict = ridge.predict(x_new)


C = (np.dot(x.T, x)+(alpha*np.identity(x.shape[1])))
W = np.dot(np.dot(np.linalg.inv(C),x.T), y)
b = np.mean(y)-W.T*np.mean(x)
y_new = W*x_new + b

#plot
plt.subplots(1, 2, figsize=(12,4))
plt.subplot(121)
plt.title("sklearn Ridge (lambda={})".format(alpha))
plt.scatter(x,y)
plt.plot(x_new, predict, 'r-', label="predict")
plt.plot(x,s, 'k-', label="original")
plt.legend()

plt.subplot(122)
plt.title("numpy Ridge (lambda={})".format(alpha))
plt.scatter(x,y)
plt.plot(x_new, y_new, 'r-', label="predict")
plt.plot(x,s, 'k-', label="original")
plt.legend()
plt.show()