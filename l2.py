# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 11:37:06 2018

@author: Admin
"""

import numpy as np
from sklearn import datasets
from sklearn.metrics import confusion_matrix, explained_variance_score
from sklearn.metrics import mean_squared_error, r2_score

print('iris data (multiclass):')
iris = datasets.load_iris()
X = iris.data
y = iris.target
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
#indices = np.random.permutation(len(X))
#print(indices)
#test_frac = 0.25
#test_size = int(test_frac * len(X))
#X_train = X[indices[:-test_size]]
#X_test = X[indices[-test_size:]]
#y_train = y[indices[:-test_size]]
#y_test = y[indices[-test_size:]]
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


print('diabetes data:')
diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target
print(np.unique(y))

X_train, X_test = make_train_test(X, 0.25)
y_train, y_test = make_train_test(y, 0.25)
print(X_train.shape)
print(X_test.shape)

print('linear regression:')
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X_train, y_train)
print('coefs:')
print(linreg.coef_)

y_pred = linreg.predict(X_test)
for i in np.random.choice(len(y_test), 10):
    print(f'{i}: actual: {y_test[i]}, prediction: {y_pred[i]:.1f}')
# explained variance score (1.0 = perfect)
print(f'score: {explained_variance_score(y_test, y_pred):.2f}')    
print(f'mean squared error: {mean_squared_error(y_test, y_pred):.2f}')
print(f'r2_score: {r2_score(y_test, y_pred):.2f}')


print('plot:')
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
bmi = [x[2] for x in X_test]
ax.plot(bmi, y_test, 'or', label='actual')
ax.plot(bmi, y_pred, '*b', label='prediction')
plt.legend()
plt.show()