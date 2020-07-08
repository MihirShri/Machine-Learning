"""
Author @ Mihir_Srivastava
Dated - 16-05-2020
File - Using_dummy_variables
Aim - Predict the housing prices given the town and area using dummy variables (columns)
"""

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn import linear_model

# Read csv file
f = pd.read_csv('homeprices one.csv')

# Get dummy variables (columns)
d = pd.get_dummies(f.town)

# Concatenate the dummy columns to original dataframe
merged = pd.concat([f, d], axis=1)

# Drop the unnecessary columns and one dummy column which can be derived from the other dummy columns
df = merged.drop(['town', 'west windsor'], axis=1)  # Dropping 'west windsor'

# Creating Linear Regression object
reg = linear_model.LinearRegression()

# Defining features and labels
X = np.array(df.drop('price', axis=1))
y = np.array(df.price)

# Training model
reg.fit(X, y)

# Finding accuracy
accuracy = reg.score(X, y)

# Predicting the final output
print(reg.predict([[2800, 0, 1]]))
