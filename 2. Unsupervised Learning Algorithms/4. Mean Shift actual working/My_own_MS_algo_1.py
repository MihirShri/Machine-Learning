"""
Author @ Mihir_Srivastava
Dated - 23-05-2020
File - My_own_MS_algo_1
Aim - To build my own mean shift algorithm from scratch given the radius.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')


# creating Mean Shift class
class Mean_Shift:
    # defining constructor
    def __init__(self, radius=4):
        self.radius = radius

    # training function
    def fit(self, data):
        # a dictionary which will contain the centroids
        centroids = {}

        # initially each data point is a centroid
        for i in range(len(data)):
            centroids[i] = data[i]

        while True:
            # this list will contain the ew centroids
            new_centroids = []

            for i in centroids:
                # this list will contain the data points that are within the radius of the centroid
                in_bandwidth = []

                centroid = centroids[i]
                for featureset in data:
                    # if it is within the radius, include it in the bandwidth
                    if np.linalg.norm(featureset - centroid) < self.radius:
                        in_bandwidth.append(featureset)

                # the new centroid is the mean of all the data points within the bandwidth
                new_centroid = np.average(in_bandwidth, axis=0)
                new_centroids.append(tuple(new_centroid))

            # since different data points might have the same centroid, hence we choose unique
            uniques = sorted(list(set(new_centroids)))

            # saving the current centroids to compare with the new centroids later
            prev_centroids = dict(centroids)

            # a dictionary to contain the new centroids
            centroids = {}
            for i in range(len(uniques)):
                centroids[i] = uniques[i]

            optimized = True

            for i in centroids:
                # we compare if our previously generated centroids are equal to the newly generated ones
                if not np.array_equal(centroids[i], prev_centroids[i]):
                    optimized = False
                if not optimized:
                    break

            if optimized:
                break
        self.centroids = centroids

        self.classifications = {}

        for i in range(len(self.centroids)):
            self.classifications[i] = []

        for featureset in data:
            # compare distance to either centroid
            distances = [np.linalg.norm(featureset - self.centroids[centroid]) for centroid in self.centroids]
            classification = (distances.index(min(distances)))

            # featureset that belongs to that cluster
            self.classifications[classification].append(featureset)

    def predict(self, data):
        # compare distance to either centroid
        distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
        classification = (distances.index(min(distances)))
        return classification


X = np.array([[1, 2],
              [1.5, 1.8],
              [5, 8],
              [8, 8],
              [1, 0.6],
              [9, 11],
              [8, 2],
              [10, 2],
              [9, 3]])

colors = 10 * ['r', 'g', 'b', 'c', 'k', 'y']

# creating Mean Shift class object
ms = Mean_Shift()

# train
ms.fit(X)

# getting centroids
centroids = ms.centroids

# plotting the data points
for classification in ms.classifications:
    color = colors[classification]
    for featureset in ms.classifications[classification]:
        plt.scatter(featureset[0], featureset[1], marker="x", color=color, s=150, linewidths=50, zorder=10)


# plotting centroids
for c in centroids:
    plt.scatter(centroids[c][0], centroids[c][1], color='k', marker='*', s=100)

plt.show()
