"""
Author @ Mihir_Srivastava
Dated - 15-05-2020
File - Predicting_housing_prices
Aim - To predict the housing price, given the area of the house using Linear Regression in one variable.
"""

# Import the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, model_selection  # Importing model_selection just to check accuracy of our model

# read the csv file
df = pd.read_csv('homeprices.csv')

# Visualize the data with the help of a scatter plot
plt.xlabel('area')
plt.ylabel('price')
plt.scatter(df['area'], df['price'], marker='+', color='red')
plt.show()

# create a LinearRegression object named 'reg'
reg = linear_model.LinearRegression()

# Specify the features (X) and labels (y). Note that in leyman language, features are the input to our model while
# labels are the output we want
X = np.array(df[['area']])  # features have to be in the form of a 2-D array.
y = np.array(df['price'])  # label can be 1-D

# Split the data (features and labels) into 2 parts, namely, the 'Training set' and the 'Testing set'.
# In our case we take the 'Testing set' to be 30% of the length of our dataframe.
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3)

# fit is used for training data
reg.fit(X_train, y_train)

# score is used for testing data
accuracy = reg.score(X_test, y_test)  # This gives the accuracy of our model

# In this particular code, I don't want to split my data into training and testing sets
reg.fit(X, y)

# this statement can be used to predict the price of a house with the specified area
print(reg.predict([[3300]]))

# Suppose we have a list of areas in a csv file and we want to predict the prices for all those areas and send it
# back to the csv file

# Read the csv file containing the list of areas for which pricing needs to be predicted
d = pd.read_csv('areas.csv')

# Store the predicted prices in a separate column
d['prices'] = reg.predict(d)

# Send it to an output file
d.to_csv('prediction.csv')

# Visualization of the line, price = m * area + b
plt.xlabel('area')
plt.ylabel('price')
plt.scatter(df['area'], df['price'], marker='+', color='red')
plt.plot(df.area, reg.predict(df[['area']]), color='blue')
plt.show()
