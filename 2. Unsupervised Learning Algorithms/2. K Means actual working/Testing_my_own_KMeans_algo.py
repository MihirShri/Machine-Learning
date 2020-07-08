"""
Author @ Mihir_Srivastava
Dated - 22-05-2020
File - Testing_my_own_KMeans_algo
Aim - To test my own K-Means-clustering algorithm on the titanic DataSet.
"""

import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import pandas as pd


# Creating a class for our algorithm
class Kmeans:
    # Defining constructor
    def __init__(self, k=2, tol=0.001, max_iter=3000):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    # Defining a function to train our model
    def fit(self, data):
        # A dictionary which will contain the centroids
        self.centroids = {}

        # Populating centroid dictionary by initially choosing the first k points as centroids
        for i in range(self.k):
            self.centroids[i] = data[i]

        for i in range(self.max_iter):
            # classifications dictionary contains the centroid index as their key and the corresponding minimum
            # distanced points from that centroid as their values
            self.classifications = {}
            for i in range(self.k):
                self.classifications[i] = []
                # Populating the classifications dictionary with the minimum distanced points/features
            for featureset in data:
                distances = [np.linalg.norm(featureset - self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)

            # This is to know what the previous centroid was to calculate the difference with the next centroid and
            # check with tolerance.
            prev_centroid = dict(self.centroids)

            # Updating centroid
            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)

            optimized = True

            # Checking whether centroid stopped changing or not
            for c in self.centroids:
                original_centroid = prev_centroid[c]
                current_centroid = self.centroids[c]
                if abs(np.sum((current_centroid - original_centroid) / original_centroid * 100.0)) > self.tol:
                    optimized = False

            # If it stopped changing then break
            if optimized:
                break

    # Function for predicting
    def predict(self, data):
        distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification


df = pd.read_excel('titanic.xls')

# Drop unnecessary columns
df.drop(['name', 'body'], 1, inplace=True)

# Fill in the missing values
df.fillna(0, inplace=True)

# Create LabelEncoder object
le = LabelEncoder()

# Perform Label Encoding on the required columns
for i in df.columns.values:
    if df[i].dtype != np.int64 and df[i].dtype != np.float64:
        df[i] = le.fit_transform(df[i].astype(str))

# Creating KMeans object with k = 2 (default)
km = Kmeans()

X = np.array(df.drop(['survived'], 1).astype(float))
X = preprocessing.scale(X)
y = np.array(df['survived'])

# Train
km.fit(X)

correct = 0

# Comparing the predicted and actual values
for i in range(len(X)):

    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = km.predict(predict_me)
    if prediction == y[i]:
        correct += 1

# Finding accuracy
accuracy = correct / len(X)
print(max(accuracy, 1 - accuracy))
