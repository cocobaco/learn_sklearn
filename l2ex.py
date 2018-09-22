# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 19:53:04 2018

@author: Admin
"""

import numpy as np
from sklearn import datasets
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

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


print('k-nearest neighbors classifier:')
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
for i in np.random.choice(len(y_test), 10):
    print(f'{i}: actual: {y_test[i]}, prediction: {y_pred[i]}')    
print(confusion_matrix(y_test, y_pred))    
print(classification_report(y_test, y_pred))    
