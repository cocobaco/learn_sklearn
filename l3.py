# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 07:35:26 2018

@author: Admin
"""

# scores and cross validataion

import numpy as np
from sklearn import datasets
from sklearn.model_selection import cross_val_score

print('digits data (multiclass):')
digits = datasets.load_digits()
X = digits.data
y = digits.target
print(np.unique(y))

print('split into train/test:')

def make_train_test(x, test_frac, seed=77):
    np.random.seed(seed)
    indices = np.random.permutation(len(x))
    test_size = int(test_frac * len(x))
    x_tr = x[indices[:-test_size]]
    x_ts = x[indices[-test_size:]]
    return x_tr, x_ts


X_train, X_test = make_train_test(X, 0.25)
y_train, y_test = make_train_test(y, 0.25)
print(X_train.shape)
print(X_test.shape)


print('support vector machine:')
from sklearn.svm import SVC
svc = SVC(kernel='linear')
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
print('score:', svc.score(X_test, y_test))
nfolds = 5
print('cross val score:', cross_val_score(svc, X, y, cv=nfolds))


print('lasso:')
from sklearn.linear_model import LogisticRegression
lass = LogisticRegression(C=1.0, penalty='l1')
lass.fit(X_train, y_train)
y_pred = lass.predict(X_test)
print('score:', lass.score(X_test, y_test))
print('cross val score:', cross_val_score(lass, X, y, cv=nfolds))


print('grid search to optimize parameters:')
from sklearn.model_selection import GridSearchCV
print('svc:')
param_grid = {'C': [0.01, 0.1, 1, 1.5, 2],  # Inverse of regularization strength
              'kernel': ['rbf', 'linear', 'poly'], 
              'shrinking': [True, False]}
grid_search = GridSearchCV(svc, 
                           param_grid=param_grid, 
                           cv=nfolds)
grid_search.fit(X, y)
print('best params:')
print(grid_search.best_params_)
# summarize the results of the grid search
print("Best score: %0.2f%%" % (100 * grid_search.best_score_))
print("Best parameter C: %f" % (grid_search.best_estimator_.C))

print('lasso:')
param_grid = {'C': np.logspace(-2, 2, 5)}
grid_search = GridSearchCV(lass, 
                           param_grid=param_grid, 
                           cv=nfolds)
grid_search.fit(X, y)
print('best params:')
print(grid_search.best_params_)
# summarize the results of the grid search
print("Best score: %0.2f%%" % (100 * grid_search.best_score_))
print("Best parameter C: %f" % (grid_search.best_estimator_.C))