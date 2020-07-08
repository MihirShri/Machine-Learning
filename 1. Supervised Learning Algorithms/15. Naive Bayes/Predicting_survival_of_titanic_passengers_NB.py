"""
Author @ Mihir_Srivastava
Dated - 23-05-2020
File - Predicting_survival_of_titanic_passengers_NB
Aim - To predict whether a passenger on titanic survived or not based on the given data using Naive Bayes algorithm
"""

# import necessary libraries
import pandas as pd
import numpy as np
import math
from sklearn import naive_bayes
from sklearn.model_selection import train_test_split

# Read csv file
df = pd.read_csv('titanic1.csv')

# Use the following code snippet to see data of all columns in pycharm (it actually displays '...' after 2 columns if
# there are more than 5 columns to display)
desired_width = 320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', 15)

# doing some data cleaning
df.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked'], 1, inplace=True)
df['Age'].fillna(math.ceil(df['Age'].mean()), inplace=True)

# Get dummy variables (columns)
d = pd.get_dummies(df['Sex'])

# Concatenate the dummy columns to original dataframe
merged = pd.concat([df, d], axis=1)

# Drop the unnecessary columns and one dummy column which can be derived from the other dummy columns
df = merged.drop(['Sex'], axis=1)  # Dropping 'Sex'

print(df)

# Create Linear Regression object
model = naive_bayes.GaussianNB()

# Define features and labels
X = np.array(df.drop('Survived', axis=1))
y = np.array(df['Survived'])

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Training the model
model.fit(X_train, y_train)

# Finding accuracy by testing the model
accuracy = model.score(X_test, y_test)

print(accuracy)
