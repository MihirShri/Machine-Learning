"""
Author @ Mihir_Srivastava
Dated - 18-05-2020
File - Comparing_accuracies_of_KNN_models
Aim - Comparing the accuracies of my own KNN algorithm against Scikit learn's KNN algorithm.
"""

# import necessary libraries
import pandas as pd
import numpy as np
import warnings
import random
from sklearn.model_selection import train_test_split
from sklearn import neighbors
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


accuracies = []

for i in range(25):
    # Read csv file
    df = pd.read_csv('breast-cancer-wisconsin.data')

    # Filling in missing values
    df.replace('?', -99999, inplace=True)

    # Dropping unnecessary columns
    df.drop(['id'], axis=1, inplace=True)

    # Converting the dataframe to a list of lists and shuffling the values each time so that our train and test set
    # contain different values each time we run the code
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

    accuracies.append(correct / total)


print("Accuracy of our model: ", sum(accuracies) / len(accuracies))


accuracies1 = []

for i in range(25):
    # Read csv file
    df = pd.read_csv('breast-cancer-wisconsin.data')

    # Use the following code snippet to see data of all columns in pycharm (it actually displays '...' after 2 columns
    # if there are more than 5 columns to display)
    desired_width = 320
    pd.set_option('display.width', desired_width)
    np.set_printoptions(linewidth=desired_width)
    pd.set_option('display.max_columns', 12)

    # Filling in missing values
    df.replace('?', -99999, inplace=True)

    # Dropping unnecessary columns
    df.drop(['id'], axis=1, inplace=True)

    # Create KNN object
    model = neighbors.KNeighborsClassifier()

    # Define features and labels
    X = np.array(df.drop(['class'], axis=1))
    y = np.array(df['class'])

    # Split the data into training and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Train the model
    model.fit(X_train, y_train)

    # Find accuracy
    accuracy = model.score(X_test, y_test)

    accuracies1.append(accuracy)


print("Accuracy of sklearn's model: ", sum(accuracies1) / len(accuracies1))
