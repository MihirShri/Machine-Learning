"""
Author @ Mihir_Srivastava
Dated - 22-05-2020
File - Grouping_titanic_passengers_MS
Aim - To group the titanic passengers into clusters using the Mean Shift algorithm and find out the actual number of
clusters and what affects them.
"""

# import necessary libraries
import pandas as pd
import numpy as np
from sklearn.cluster import MeanShift
from sklearn import preprocessing

# Read xls file
df = pd.read_excel('titanic.xls')

# Creating a copy of the DataFrame to be used later
original_df = pd.DataFrame.copy(df)

# Use the following code snippet to see data of all columns in pycharm (it actually displays '...' after 2 columns if
# there are more than 5 columns to display)
desired_width = 320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', 15)

# Drop unnecessary columns
df.drop(['name', 'body'], 1, inplace=True)

# Fill in the missing values
df.fillna(0, inplace=True)


# handling non-numerical data
def handle_non_numerical_data(df):
    columns = df.columns.values

    for column in columns:
        text_digit_vals = {}

        def convert_to_int(val):
            return text_digit_vals[val]

        # if data type is either int or float then don't change
        if df[column].dtype != np.int64 and df[column].dtype != np.float64:

            column_contents = df[column].values.tolist()

            # finding just the uniques
            unique_elements = set(column_contents)

            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    # creating dict that contains new id per unique string
                    text_digit_vals[unique] = x
                    x += 1
            # now we map the new "id" value to replace the string.
            df[column] = list(map(convert_to_int, df[column]))

    return df


df = handle_non_numerical_data(df)

# Creating KMeans object with k = 2
ms = MeanShift()

df.drop(['ticket', 'home.dest'], 1, inplace=True)

# Define features
X = np.array(df.drop('survived', 1).astype(float))
X = preprocessing.scale(X)

# Defining label for comparison purposes
y = np.array(df['survived'])

# Train
ms.fit(X)

# getting labels and cluster centers
labels = ms.labels_
cluster_centers = ms.cluster_centers_

# creating a new column that contains the cluster groups (labels)
original_df['cluster_group'] = pd.Series(labels)

# getting the number of clusters
n_clusters = len(np.unique(labels))

# this dictionary will contain the survival rates of the different clustering groups.
survival_rates = {}
for i in range(n_clusters):
    temp_df = original_df[original_df['cluster_group'] == float(i)]
    survival_cluster = temp_df[temp_df['survived'] == 1]
    survival_rate = len(survival_cluster) / len(temp_df)
    survival_rates[i] = survival_rate

print(survival_rates)

# print(original_df[original_df['cluster_group'] == 0])
