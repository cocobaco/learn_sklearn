# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 11:37:06 2018

@author: Admin
"""

import numpy as np
from sklearn import datasets
from sklearn.metrics import classification_report
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


print('logistic regression:')
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))    


print('support vector machine:')
from sklearn.svm import SVC
svc = SVC(kernel='linear')
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))


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


print('ridge regression:')
from sklearn.linear_model import Ridge
rreg = Ridge()
rreg.fit(X_train, y_train)
y_pred = rreg.predict(X_test)
# explained variance score (1.0 = perfect)
print(f'score: {explained_variance_score(y_test, y_pred):.2f}')    
print(f'mean squared error: {mean_squared_error(y_test, y_pred):.2f}')
print(f'r2_score: {r2_score(y_test, y_pred):.2f}')


def check_alpha(mymodel, alphas=np.linspace(0.01, 0.6, num=15)):
    plt.figure()
    for a in alphas:
        print(f'alpha = {a:.2f}:')
        model = mymodel(alpha=a)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f'mean squared error: {mean_squared_error(y_test, y_pred):.2f}')
        print(f'r2_score: {r2_score(y_test, y_pred):.2f}')
        plt.plot(a, r2_score(y_test, y_pred), 'og', markersize=10)
    plt.xlabel('alpha')
    plt.ylabel('r2 score')
    
    plt.show()
    
print('varying ridge alpha:')
alphas = np.linspace(0.01, 0.5, num=10)
check_alpha(Ridge, alphas)


print('lasso regression:')
from sklearn.linear_model import Lasso
lass = Lasso()
lass.fit(X_train, y_train)
y_pred = lass.predict(X_test)
print(f'score: {explained_variance_score(y_test, y_pred):.2f}')    
print(f'mean squared error: {mean_squared_error(y_test, y_pred):.2f}')
print(f'r2_score: {r2_score(y_test, y_pred):.2f}')
print('varying lasso alpha:')
alphas = np.linspace(0.01, 0.5, num=10)
check_alpha(Lasso, alphas)