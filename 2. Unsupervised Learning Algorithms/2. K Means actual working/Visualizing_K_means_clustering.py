"""
Author @ Mihir_Srivastava
Dated - 20-05-2020
File - Visualizing_K_means_clustering
Aim - To visualize how k means clustering (flat clustering) actually works.
"""

# import necessary libraries
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
from sklearn.cluster import KMeans

style.use('ggplot')

# create data set
X = np.array([[1, 2],
              [1.5, 1.8],
              [5, 8],
              [8, 8],
              [1, 0.6],
              [9, 11]])

# Create Kmeans object and specify the number of clusters to be 2
clf = KMeans(n_clusters=2)
clf.fit(X)

centroids = clf.cluster_centers_

# Labels can be thought of as synonymous to "y" in supervised learning
labels = clf.labels_

# For specifying the colors of different clusters
colors = ["g.", "r.", "c.", "y."]

# Visualizing the clusters
for i in range(len(X)):
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize=10)
plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=75, linewidths=5, zorder=10)
plt.show()
