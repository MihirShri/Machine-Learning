"""
Author @ Mihir_Srivastava
Dated - 15-05-2020
File - Train_Test_Split
Aim - To train a model and test it.
"""

# import necessary libraries
import pandas as pd
import numpy as np
import math
from sklearn import model_selection
from sklearn.linear_model import LinearRegression

# Read csv file
df = pd.read_csv('carprices train test.csv')

# Renaming columns as per need
df.rename(columns={'Age(yrs)': 'Age', 'Sell Price($)': 'Price'}, inplace=True)

# Create Linear Regression object
reg = LinearRegression()

# Define features and labels
X = np.array(df[['Mileage', 'Age']])
y = np.array(df['Price'])

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

# Training the model
reg.fit(X_train, y_train)

# Finding accuracy by testing the model
accuracy = reg.score(X_test, y_test)

print(accuracy)
