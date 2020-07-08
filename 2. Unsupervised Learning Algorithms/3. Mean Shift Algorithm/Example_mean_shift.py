"""
Author @ Mihir_Srivastava
Dated - 22-05-2020
File - Example_mean_shift
Aim - An example which shows the working of Mean Shift algorithm.
"""

# import required libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.cluster import MeanShift
from sklearn.datasets import make_blobs

style.use('ggplot')

# Defining centers around which sample data is to be created
centres = [[1, 1, 1], [5, 5, 5], [3, 10, 10]]

# make sample data with the specified centers, 100 samples and standard deviation = 1
X, _ = make_blobs(n_samples=100, centers=centres, cluster_std=1)

# Create MeanShift object
ms = MeanShift()

# Train
ms.fit(X)

# getting labels
labels = ms.labels_

# getting the new cluster centers
cluster_centres = ms.cluster_centers_
print(cluster_centres)

# getting the number of clusters
n_clusters = len(np.unique(labels))
print("Number of estimated clusters: ", n_clusters)

# defining colors to give to different clusters
colors = 10*['r', 'b', 'g']
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Visualizing
for i in range(len(X)):
    ax.scatter(X[i][0], X[i][1], X[i][2], c=colors[labels[i]], marker='o')

ax.scatter(cluster_centres[:, 0], cluster_centres[:, 1], cluster_centres[:, 2], marker='x', color='k', s=150,
           linewidths=5, zorder=10)

plt.show()
