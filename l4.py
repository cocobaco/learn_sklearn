# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 14:28:25 2018

@author: Admin
"""

# clustering

import numpy as np
from sklearn import datasets
from sklearn.cluster import KMeans

iris = datasets.load_iris()
X = iris.data
y = iris.target
print(np.unique(y))

kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

for i in np.random.choice(len(y), 10):
    print(f'{i}: actual: {y[i]}, prediction: {kmeans.labels_[i]}')
    
for i in np.arange(0, len(y), 10):
    print(f'{i}: actual: {y[i]}, prediction: {kmeans.labels_[i]}')
    
print('varying n_clusters:')
nclus = np.arange(2, 8)
inertias = {}
for n in nclus:
    kmeans = KMeans(n_clusters=n)
    kmeans.fit(X)
    inertias[n] = kmeans.inertia_
    

import matplotlib.pyplot as plt

plt.figure()
plt.bar(x=np.arange(len(inertias)), height=list(inertias.values()), 
        align='center')
plt.xticks(np.arange(len(inertias)), list(inertias.keys()))
plt.xlabel('n clusters')
plt.ylabel('inertia')

plt.figure()
plt.scatter(list(inertias.keys()), list(inertias.values()), 
            alpha=0.6, c='g', s=100, marker=r'$\alpha$')
#plt.xticks(np.arange(len(inertias)), list(inertias.keys()))
plt.xlabel('n clusters')
plt.ylabel('inertia')

plt.show()


print('*' * 50)
print('PCA:')
x1 = np.random.normal(size=100)
x2 = np.random.normal(size=100)
x3 = x1 + x2
xc = np.c_[x1, x2, x3]
from sklearn.decomposition import PCA
pca = PCA()
pca.fit(xc)
print('explained variance:')
print(pca.explained_variance_)
# only the 2 first components are useful
pca = PCA(n_components=2)
xc_reduced = pca.fit_transform(xc)
print(pca.explained_variance_)

