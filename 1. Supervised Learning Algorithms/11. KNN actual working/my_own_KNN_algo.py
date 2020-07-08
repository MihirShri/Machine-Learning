"""
Author @ Mihir_Srivastava
Dated - 18-05-2020
File - my_own_KNN_algorithm
Aim - Creating the KNN algorithm on my own.
"""

# import necessary libraries
import numpy as np
import warnings
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')

# Defining our dataset
dataset = {'k': [[1, 2], [2, 3], [3, 1]], 'r': [[6, 5], [7, 7], [8, 6]]}

# The point whose class needs to be predicted
new_feature = [4, 4]


# Defining the core function of our algorithm
def k_nearest_neighbors(data, predict, k=3):
    # K should always be greater than the number of classes
    if len(data) >= k:
        warnings.warn('Idiot!')

    distances = []

    for group in data:
        for feature in data[group]:
            # Calculating the euclidean distance
            euclidean_distance = np.linalg.norm(np.array(feature) - np.array(predict))
            distances.append([euclidean_distance, group])

    # We just want the groups and not the euclidean distance hence i[1]. (i[0] corresponds to euclidean distance)
    votes = [i[1] for i in sorted(distances)[:k]]

    # most_common returns a list of tuples where the first value corresponds to the actual class while the second
    # value is the number of times it occurs
    vote_result = Counter(votes).most_common(1)[0][0]

    return vote_result


result = k_nearest_neighbors(dataset, new_feature, k=3)
print(result)

# Visualization of the dataset and point
[[plt.scatter(ii[0], ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
plt.scatter(new_feature[0], new_feature[1], color=result, marker='+', s=50)
plt.show()
