# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 17:14:47 2019

@author: 321
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#data
n = 100
x=np.linspace(-3,3,n)
x=np.reshape(x, (-1,1))
s = 80*x - 100
s=s.reshape(n,1)
num =70*np.random.randn(n,1)
y =s+num

#training
lr = LinearRegression().fit(x, y)
print("lr.intercept_: {}".format(lr.intercept_))
print("lr.coef_: {}".format(lr.coef_))

x_new = np.linspace(-3,3,n)
x_new=np.reshape(x_new, (-1,1))

predict = lr.predict(x_new)


X = np.hstack([np.ones((x.shape[0], 1)), x])
theta = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)
print("theta: {}".format(theta))

X_new = np.hstack([np.ones((x_new.shape[0], 1)), x_new])
y_new = np.dot(X_new, theta)

#plot
plt.subplots(1,2, figsize=(12,4))
plt.subplot(1,2,1)
plt.title("scikit predict")
plt.scatter(x,y)
plt.plot(x_new, predict, 'r-', label="predict")
plt.plot(x,s, 'k-', label="original")
plt.legend()

plt.subplot(1,2,2)
plt.title("numpy result")
plt.scatter(x,y)
plt.plot(x_new, y_new, 'r-', label="predict")
plt.plot(x,s, 'k-', label="original")
plt.legend()
plt.show()
