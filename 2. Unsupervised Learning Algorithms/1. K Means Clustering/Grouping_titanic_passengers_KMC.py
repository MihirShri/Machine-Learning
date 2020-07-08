"""
Author @ Mihir_Srivastava
Dated - 20-05-2020
File - Grouping_titanic_passengers_KMC
Aim - To group the titanic passengers into two clusters and check the accuracy by comparing it with the 'survived'
column (which tells whether the passenger survived or not). Our main goal is to check whether there is a clear
distinction between the two groups, survived and not survived, based on the available data.
"""

# import necessary libraries
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing

# Read xls file
df = pd.read_excel('titanic.xls')

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
km = KMeans(n_clusters=2)

# Define features
X = np.array(df.drop('survived', 1).astype(float))
X = preprocessing.scale(X)

# Defining label for comparison purposes
y = np.array(df['survived'])

# Train
km.fit(X)

correct = 0

# Comparing the predicted and actual values
for i in range(len(X)):
    if y[i] == km.labels_[i]:
        correct += 1

# Finding accuracy
accuracy = correct / len(X)
print(max(accuracy, 1 - accuracy))
