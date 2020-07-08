"""
Author @ Mihir_Srivastava
Dated - 15-05-2020
File - Predicting_income_per_capita
Aim - To predict the per capita income of Canada for the year 2020 using Linear Regression in one variable.
"""

# Import the required libraries
import pandas as pd
import numpy as np
from sklearn import linear_model

# Read the csv file containing data
df = pd.read_csv('canada_per_capita_income.csv')

# Some data cleaning
df = df.drop(df.columns[[2]], axis=1)
df.rename(columns={'per capita income (US$)': 'per capita income'}, inplace=True)

# Creating a LinearRegression object
reg = linear_model.LinearRegression()

# Defining features and labels
X = np.array(df[['year']])
y = np.array(df['per capita income'])

# Training the model
reg.fit(X, y)

# Predicting the per capita income of Canada for the year 2020
print(reg.predict([[2020]]))
