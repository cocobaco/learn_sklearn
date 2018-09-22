# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 11:37:06 2018

@author: Admin
"""

import numpy as np
from sklearn import datasets

iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target
print(np.unique(iris_y))
