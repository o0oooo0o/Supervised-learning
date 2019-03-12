# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 14:12:00 2019

@author: 321
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#data
mu_c = np.array([2, 6])
mu_m = np.array([4, 5])
sigma = np.array([[1, 1.5], [1.5, 3]])
cyan = np.random.multivariate_normal(mu_c, sigma, 50)
magenta = np.random.multivariate_normal(mu_m, sigma, 50)

x = np.concatenate([cyan, magenta])
y = np.concatenate([np.zeros(50), np.ones(50)])

#LDA
lda = LinearDiscriminantAnalysis(n_components=2).fit(x,y)

n =500
X1, X2 = np.meshgrid(np.linspace(-1, 7, n), np.linspace(1, 10, n))
XX = np.c_[X1.ravel(), X2.ravel()]

predict = lda.predict(XX).reshape(X1.shape)


prior_c = len(cyan) / len(x)
prior_m = len(magenta) / len(x)

def boundary(data):
    mu = (mu_c - mu_m).reshape(2,1)
    A = np.dot(mu.T, np.linalg.inv(sigma))
    b = -0.5*np.dot(A, (mu_c+mu_m))-np.log(prior_c/prior_m)
    
    if np.dot(A, data)+b > 0:
        result = 1
    else:
        result = 0
    
    return result

results = []
for i in range(0,n*n):
    results.append(boundary(XX[i].reshape(2,1)))
    
results = np.array(results).reshape(n,n)

#plot
plt.subplots(1,2,figsize=(12,4))
plt.subplot(121)
plt.title("sklearn LDA")
plt.scatter(x[y == 0, 0], x[y == 0, 1], color='c')
plt.scatter(x[y == 1, 0], x[y == 1, 1], color='m')
plt.contour(X1, X2, predict, 0, colors='k')
plt.xlim(-1, 7)
plt.ylim(1, 10)

plt.subplot(122)
plt.title("numpy LDA")
plt.scatter(x[y == 0, 0], x[y == 0, 1], color='c')
plt.scatter(x[y == 1, 0], x[y == 1, 1], color='m')
plt.contour(X1, X2, results, 0, colors='k')
plt.xlim(-1, 7)
plt.ylim(1, 10)
plt.show()