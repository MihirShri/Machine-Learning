"""
Author @ Mihir_Srivastava
Dated - 20-05-2020
File - Grouping_iris_flower_KMC
Aim - To group iris flower into clusters (classes) based on their petal width and length using K-Means-Clustering (KMC)
algorithm by first choosing a suitable value of k using elbow method.
"""

# import necessary libraries
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

# create load_iris object
iris = load_iris()

# performing data cleaning
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df.drop(['sepal length (cm)', 'sepal width (cm)'], axis=1, inplace=True)

# specify the range of k to predict its value
k_range = range(1, 11)

# creating an empty list to store Sum of Squared Errors (SSE) for different values of k
sse = []

# this loop is to train the model for different values of k
for k in k_range:
    km = KMeans(n_clusters=k)
    km.fit(df)
    sse.append(km.inertia_)  # km.inertia_ returns the sum of squared errors for a specific value of k

# Visualizing the elbow plot to choose a suitable value of k
plt.xlabel('K')
plt.ylabel('SSE')
plt.plot(k_range, sse)
plt.grid(True)
plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
plt.show()

# Plot shows elbow bends at k = 3

# Creating KMeans object with k = 3
km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(df)
df['clusters'] = y_predicted

# Creating new DataFrames for visualization purposes based on clustering groups.
df1 = df[df['clusters'] == 0]
df2 = df[df['clusters'] == 1]
df3 = df[df['clusters'] == 2]

# Visualizing the clusters and the centroids.
plt.scatter(df1['petal length (cm)'], df1['petal width (cm)'], color='r', label='cluster 1')
plt.scatter(df2['petal length (cm)'], df2['petal width (cm)'], color='b', label='cluster 2')
plt.scatter(df3['petal length (cm)'], df3['petal width (cm)'], color='g', label='cluster 3')
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], color='k', marker='*', label='centroid')
plt.legend()
plt.show()
