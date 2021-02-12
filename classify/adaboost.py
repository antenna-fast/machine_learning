from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification

# 生成样本
X, y = make_classification(n_samples=1000, n_features=4,
                           n_informative=2, n_redundant=0,
                           random_state=0, shuffle=False)
# y+=1
print(X.shape)
print(y.shape)
print(y)

clf = AdaBoostClassifier(n_estimators=120, random_state=0)
clf.fit(X, y)

res = clf.predict([[0, 0, 0, 0]])

# Return the mean accuracy on the given test data and labels.
score = clf.score(X, y)

print('res:', res)
print('score:', score)
