# http://gael-varoquaux.info/scikit-learn-tutorial/settings.html

from sklearn import datasets

# iris flowers
iris = datasets.load_iris()
print(type(iris))
print(dir(iris))
iris_data = iris.data
iris_target = iris.target
print(iris.feature_names)
print(iris_data.shape)
print(type(iris_data))

print(dir())
#print(globals())
#print(locals())

# handwritten digits
digits = datasets.load_digits()
print(dir(digits))
print(digits.data.shape)
print(digits.images.shape)
digits_data = digits.data
digits_target = digits.target

'''
When the data is not intially in the (n_samples, n_features) shape, 
it needs to be preprocessed to be used by the scikit.
'''
nrow = digits.images.shape[0]
digits_images = digits.images.reshape((nrow, -1))

print(digits_target)

import pylab as pl

pl.imshow(digits.images[0])

pl.imshow(digits.images[9], cmap=pl.cm.gray_r)

print(dir())
#print(globals())
#print(locals())

