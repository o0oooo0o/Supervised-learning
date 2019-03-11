# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 15:23:28 2019

@author: 321
"""

import matplotlib.pyplot as plt
import numpy as np

#datda
x = np.concatenate([np.random.normal(4, 1, 200), np.random.normal(10, 2, 1000)])
y = np.concatenate([np.zeros(200), np.ones(1000)])

x1 = x[:200]
x2 = x[200:]

#bayes classifier
prior_x1 = len(x1) / len(x)
prior_x2 = len(x2) / len(x)

xx = np.linspace(0, 15, 700)

def likelihood(data, idxs):
    mu = np.mean(data)
    std = np.std(data)
    
    result = []
    for i in idxs:
        exponent = np.exp(-((i-mu)**2 / (2*(std**2))))
        result.append(exponent / (np.sqrt(2*np.pi) * std))
    
    result = np.array(result)
    return result

post_x1 = likelihood(x1, xx) * prior_x1
post_x2 = likelihood(x2, xx) * prior_x2

prob_x1 = post_x1 / (post_x1 + post_x2)
prob_x2 = post_x2 / (post_x1 + post_x2)

#plot
plt.subplots(3,1, figsize=(12,8))
plt.subplot(311)
plt.hist(x1, bins=25, color='m', alpha=.7)
plt.hist(x2, bins=25, color='c', alpha=.7)
plt.ylabel("Count")

plt.subplot(312)
plt.plot(xx, post_x1, color='m')
plt.plot(xx, post_x2, color='c')
plt.ylabel("P(M)P(D|M)")

plt.subplot(313)
plt.plot(xx, prob_x1, color='m')
plt.plot(xx, prob_x2, color='c')
plt.ylabel("P(M|D)")
plt.show()