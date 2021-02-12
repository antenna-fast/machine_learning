from numpy import *
from sklearn import svm

# SVM分类器

# 用于对特征分类，得到特征对应点的类别后，

# X = [[0, 0],
#      [1, 1]]
X = [zeros(33),
    ones(33)]

y = [0, 1]  # 类别标签


# 自定义核函数
def my_kernel(X, Y):
    return dot(X, Y.T)


# kernel='linear'
clf = svm.SVC(kernel='rbf')
clf.fit(X, y)

data_pred = ones(33)
# data_pred = ones(33)

# res = clf.predict([[2., 2.]])
res = clf.predict([data_pred])
print('res:', res)

# 支撑向量
# clf.support_vectors_

# get indices of support vectors
# clf.support_

# get number of support vectors for each class
# clf.n_support_
#
