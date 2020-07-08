"""
Author @ Mihir_Srivastava
Dated - 16-05-2020
File - Save_using_pickle
Aim - To train a Linear Regression model based on given data and save it in a file using joblib
"""

# Import the required libraries
import pandas as pd
import numpy as np
from sklearn import linear_model
import joblib

# read the csv file
df = pd.read_csv('homeprices.csv')

# create a LinearRegression object named 'reg'
reg = linear_model.LinearRegression()

# Defining features and labels
X = np.array(df[['area']])
y = np.array(df['price'])

# Training model
reg.fit(X, y)

# Predicting output
print(reg.predict([[3300]]))

# Save the trained model in a binary file
joblib.dump(reg, 'model_joblib')
