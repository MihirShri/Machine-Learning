"""
Author @ Mihir_Srivastava
Dated - 17-05-2020
File - Predicting_salary_decision_tree
Aim - To predict whether the salary of an employee is greater than 100k or not based on his company, job position and degree
"""

# import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.preprocessing import LabelEncoder

# Read csv file
df = pd.read_csv('salaries.csv')

# Use the following code snippet to see data of all columns in pycharm (it actually displays '...' after 2 columns if
# there are more than 5 columns to display)
desired_width = 320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', 10)

# Create LabelEncoder object
le = LabelEncoder()

# Perform label encoding
df['company_n'] = le.fit_transform(df['company'])
df['job_n'] = le.fit_transform(df['job'])
df['degree_n'] = le.fit_transform(df['degree'])

# Printing just to get an idea of which labels to pass for prediction
print(df)

# Dropping unnecessary columns
df = df.drop(['company', 'job', 'degree'], axis=1)

# Create decision tree object
model = tree.DecisionTreeClassifier()

# Define features and labels
X = df.drop(['salary_more_then_100k'], axis=1)
y = df.salary_more_then_100k

# Split the data into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train the model
model.fit(X_train, y_train)

# Find accuracy
accuracy = model.score(X_test, y_test)
print("accuracy: " + str(accuracy))

# Make prediction
predict = model.predict([[0, 2, 0]])

if predict == 1:
    print("Salary > 100k")
else:
    print("Salary < 100k")
