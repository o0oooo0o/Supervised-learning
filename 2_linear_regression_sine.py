# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 13:31:31 2019

@author: 321
"""
#polynomial Regression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#data
n = 50
x=np.linspace(0,1,n)
#x=np.reshape(x, (-1,1))
s = np.sin(2*np.pi*x)
s=s.reshape(n,1)
num =0.3*np.random.randn(n,1)
y =s+num

#training
d=9
X=np.vander(x,d+1)
lr = LinearRegression().fit(X, y)
print("lr.intercept_: {}".format(lr.intercept_))
print("lr.coef_: {}".format(lr.coef_))

x_new = np.linspace(0,1,n)
#x_new=np.reshape(x_new, (-1,1))
X_new = np.vander(x_new,d+1)
predict = lr.predict(X_new)


theta = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)
print("theta: {}".format(theta))

y_new = np.dot(X_new, theta)

#plot
plt.subplots(1,2, figsize=(12,4))
plt.subplot(1,2,1)
plt.title("scikit predict (d={})".format(d))
plt.scatter(x,y)
plt.plot(x, predict, 'r-', label="predict")
plt.plot(x,s, 'k-', label="original")
plt.legend()

plt.subplot(1,2,2)
plt.title("numpy result (d={})".format(d))
plt.scatter(x,y)
plt.plot(x_new, y_new, 'r-', label="predict")
plt.plot(x,s, 'k-', label="original")
plt.legend()
plt.show()
