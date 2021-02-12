import numpy as np
import pandas as pd
from sklearn.cluster import MeanShift
from sklearn.datasets.samples_generator import make_blobs
from matplotlib import pyplot as plt

# from mpl_toolkits.mplot3d import Axes3D

# 生成数据
clusters = [[1, 1, 1], [5, 5, 5], [3, 10, 10]]
X, _ = make_blobs(n_samples=150, centers=clusters, cluster_std=0.60)

ms = MeanShift()
ms.fit(X)
cluster_centers = ms.cluster_centers_  # 聚类结果

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], marker='o')
ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], cluster_centers[:, 2], marker='x', color='red', s=300,
           linewidth=5, zorder=10)
plt.show()
