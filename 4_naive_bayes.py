# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 15:12:53 2019

@author: 321
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB

#data
x = np.concatenate([np.random.normal(4, 2, size=(300, 2)), np.random.normal(8, 1, size=(100, 2))])
y = np.concatenate([np.zeros(300), np.ones(100)])

cyan = x[:300, :]
magenta = x[300:, :]

#naive bayes
clf = GaussianNB().fit(x, y)

n =1000
X1, X2 = np.meshgrid(np.linspace(0, 12, n), np.linspace(0, 13, n))
XX = np.c_[X1.ravel(), X2.ravel()]

predict = clf.predict(XX).reshape(X1.shape)


prior_c = cyan.shape[0] / x.shape[0]
prior_m = magenta.shape[0] / x.shape[0]

def likelihood(data, idxs):
    mu = np.mean(data)
    std = np.std(data)
    
    result = []
    for i in range(0,n*n):
        exponent1 = np.exp(-((idxs[i][0]-mu)**2 / (2*(std**2))))
        exponent2 = np.exp(-((idxs[i][1]-mu)**2 / (2*(std**2))))
        result.append((exponent1 / (np.sqrt(2*np.pi) * std))*(exponent2 / (np.sqrt(2*np.pi) * std)))
        
    result = np.array(result).reshape(n,n)
    return result

likelihood_c = likelihood(cyan, XX)
likelihood_m = likelihood(magenta, XX)
result = np.hstack((np.reshape(likelihood_c, (n*n,1)), np.reshape(likelihood_m, (n*n,1))))
result = np.argmax(result, axis=1).reshape(X1.shape)

'''
post_c = likelihood(cyan, XX) * prior_c
post_m = likelihood(magenta, XX) * prior_m
prob_c = post_c / (post_c, post_m)
prob_m = post_m / (post_c, post_m)
'''

#plot
plt.subplots(1,2, figsize=(15,4))
plt.subplot(121)
plt.title("sklearn GaussianNB")
plt.scatter(x[y == 0, 0], x[y == 0, 1], color='c')
plt.scatter(x[y == 1, 0], x[y == 1, 1], color='m')
plt.contour(X1, X2, predict, 0, colors='k', linewidth=.1)
plt.xlim(np.min(x[y == 0, 0])*0.2, np.max(x[y == 1, 0])*1.1)
plt.ylim(np.min(x[y == 0, 1])*0.2, np.max(x[y == 1, 1])*1.1)

plt.subplot(122)
plt.title("naive bayes")
plt.scatter(x[y == 0, 0], x[y == 0, 1], color='c')
plt.scatter(x[y == 1, 0], x[y == 1, 1], color='m')
plt.contour(X1, X2, result, 0, colors='k', linewidth=.1)
plt.xlim(np.min(x[y == 0, 0])*0.2, np.max(x[y == 1, 0])*1.1)
plt.ylim(np.min(x[y == 0, 1])*0.2, np.max(x[y == 1, 1])*1.1)
#plt.show()


'''
#A user make this code. I just brought it.

class GNB(object):
    def __init__(self):
        pass

    def fit(self, X, y):
        separated = [[x for x, t in zip(X, y) if t == c] for c in np.unique(y)]
        self.model = np.array([np.c_[np.mean(i, axis=0), np.std(i, axis=0)]
                    for i in separated])
        return self

    def _prob(self, x, mean, std):
        exponent = np.exp(- ((x - mean)**2 / (2 * std**2)))
        return np.log(exponent / (np.sqrt(2 * np.pi) * std))

    def predict_log_proba(self, X):
        return [[sum(self._prob(i, *s) for s, i in zip(summaries, x))
                for summaries in self.model] for x in X]

    def predict(self, X):
        return np.argmax(self.predict_log_proba(X), axis=1)

    def score(self, X, y):
        return sum(self.predict(X) == y) / len(y)


asdf = GNB().fit(x,y)
asdf_predict = asdf.predict(XX)
asdf_predict = np.reshape(asdf_predict, X1.shape)


plt.subplot(133)
plt.title("asdf GNB")
plt.scatter(x[y == 0, 0], x[y == 0, 1], color='c')
plt.scatter(x[y == 1, 0], x[y == 1, 1], color='m')
plt.contour(X1, X2, asdf_predict, 0, colors='k')
plt.xlim(np.min(x[y == 0, 0])*0.2, np.max(x[y == 1, 0])*1.1)
plt.ylim(np.min(x[y == 0, 1])*0.2, np.max(x[y == 1, 1])*1.1)
plt.show()
'''