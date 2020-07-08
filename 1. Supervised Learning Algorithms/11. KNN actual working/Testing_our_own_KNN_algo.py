"""
Author @ Mihir_Srivastava
Dated - 18-05-2020
File - Testing_our_own_KNN_algorithm
Aim - Testing my own KNN algorithm against the wisconsin breast cancer data set.
"""

# import necessary libraries
import pandas as pd
import numpy as np
import warnings
import random
from collections import Counter


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


# Read csv file
df = pd.read_csv('breast-cancer-wisconsin.data')

# Filling in missing values
df.replace('?', -99999, inplace=True)

# Dropping unnecessary columns
df.drop(['id'], axis=1, inplace=True)

# Converting the dataframe to a list of lists and shuffling the values each time so that our train and test set contain
# different values each time we run the code
full_data = df.astype(float).values.tolist()
random.shuffle(full_data)

# Creating our own test and train set
test_size = 0.2  # Size of test data set
test_set = {2: [], 4: []}
train_set = {2: [], 4: []}
train_data = full_data[:-int(test_size * len(full_data))]
test_data = full_data[-int(test_size * len(full_data)):]

for i in train_data:
    train_set[i[-1]].append(i[:-1])

for i in test_data:
    test_set[i[-1]].append(i[:-1])

correct = 0.0
total = 0.0

for group in test_set:
    for data in test_set[group]:
        vote = k_nearest_neighbors(train_set, data, k=5)
        if group == vote:
            correct += 1
        total += 1

print("accuracy: ", correct / total)
