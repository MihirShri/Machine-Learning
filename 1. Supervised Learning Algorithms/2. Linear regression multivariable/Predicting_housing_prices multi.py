"""
Author @ Mihir_Srivastava
Dated - 15-05-2020
File - Predicting_housing_prices using multivariate regression
Aim - To predict the housing price, given the area, number of bedrooms and age of the house using Linear Regression in
multiple variables.
"""

# Import necessary libraries
import pandas as pd
import numpy as np
import math
from sklearn import linear_model

# Read csv file
df = pd.read_csv('homeprices multi.csv')

# Fill the missing value with the median value
median_bedroom = math.floor(df.bedrooms.median())
df.bedrooms.fillna(median_bedroom, inplace=True)

# Create Linear Regression object
reg = linear_model.LinearRegression()

# Define features and labels
X = np.array(df[['area', 'bedrooms', 'age']])
y = np.array(df['price'])

# Train the model
reg.fit(X, y)

# Predict the output
print(reg.predict([[2500, 4, 5]]))

