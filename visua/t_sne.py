import numpy as np
from sklearn.manifold import TSNE

X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
X_embedded = TSNE(n_components=2).fit_transform(X)
print(X_embedded.shape)

# X_embedded = TSNE(n_components=3).fit_transform(fpfh_data.T)
# print(X_embedded.shape)
#
# X_embedded = X_embedded.T
# plt.scatter(X_embedded[0], X_embedded[1])
#
# plt.show()