"""
Author @ Mihir_Srivastava
Dated - 16-05-2020
File - Using_dummy_variables exercise
Aim - Predict the car prices given the mileage, age and car model using dummy variables (columns)
"""

# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

# Read csv file
f = pd.read_csv('carprices.csv')

# Renaming some columns as per need
f.rename(columns={'Sell Price($)': 'Price', 'Age(yrs)': 'Age'}, inplace=True)

# Plotting age vs price to check which ML model is applicable
plt.title('Age vs Price')
plt.xlabel('Age')
plt.ylabel('Price')
plt.scatter(f.Age, f.Price, color='red', marker='+')
plt.show()

# Plotting mileage vs price to check which ML model is applicable
plt.title('Mileage vs Price')
plt.xlabel('Mileage')
plt.ylabel('Price')
plt.scatter(f.Mileage, f.Price, color='red', marker='+')
plt.show()

# Get dummy variables (columns)
d = pd.get_dummies(f['Car Model'])

# Concatenate the dummy columns to original dataframe
merged = pd.concat([f, d], axis=1)

# Drop the unnecessary columns and one dummy column which can be derived from the other dummy columns
df = merged.drop(['Car Model', 'Mercedez Benz C class'], axis=1)  # Dropping 'Mercedez Benz C class'

# Creating Linear Regression object
reg = linear_model.LinearRegression()

# Defining features and labels
X = np.array(df.drop('Price', axis=1))
y = np.array(df.Price)

# Training model
reg.fit(X, y)

# Finding accuracy
accuracy = reg.score(X, y)

# Printing required results
print(df)
print(accuracy)
print(reg.predict([[45000, 4, 0, 0]]))
print(reg.predict([[86000, 7, 0, 1]]))
