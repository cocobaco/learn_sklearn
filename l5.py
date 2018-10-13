# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 20:38:26 2018

@author: Admin
"""
# http://scikit-learn.org/0.18/auto_examples/plot_digits_pipe.html

from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import numpy as np

digits = load_digits()
X = digits.data
y = digits.target
print(X.shape)

model = LogisticRegression()

# unsupervised dimensionality reduction
pca = PCA()

pca.fit(X, y)

plt.figure()
plt.plot(pca.explained_variance_)
plt.xlabel('n_components')
plt.ylabel('explained_variance_')
plt.show()

pipe = Pipeline(steps=[('pca', pca), ('logistic', model)])

n_components = [20, 40, 64]
Cs = np.logspace(-4, 4, 3)
estimator = GridSearchCV(pipe, 
                         dict(pca__n_components=n_components, 
                              logistic__C=Cs))
estimator.fit(X, y)
print(estimator.best_params_)
plt.axvline(estimator.best_estimator_.named_steps['pca'].n_components,
            linestyle=':', label='n_components chosen')
plt.legend(prop=dict(size=12))
plt.show()