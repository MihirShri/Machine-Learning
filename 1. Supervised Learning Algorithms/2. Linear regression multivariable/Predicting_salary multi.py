"""
Author @ Mihir_Srivastava
Dated - 15-05-2020
File - Predicting_salary using multivariate regression
Aim - To predict the salary of an employee, given his experience, test score and interview score using Linear Regression
in multiple variables.
"""

# Import necessary libraries
import pandas as pd
import numpy as np
import math
from sklearn import linear_model
from word2number import w2n

# Read csv file
df = pd.read_csv('hiring.csv')

# Filling NaNs with 0
df.experience.fillna('zero', inplace=True)

# Converting experience given in words to number
df.experience = df.experience.apply(w2n.word_to_num)

# Renaming some columns
df.rename(columns={'test_score(out of 10)': 'test_score', 'interview_score(out of 10)': 'interview_score',
                   'salary($)': 'salary'}, inplace=True)

# Filling NaN with median value
df.test_score.fillna(math.floor(df.test_score.median()), inplace=True)

# Creating Linear Regression object
reg = linear_model.LinearRegression()

# Defining features and labels
X = np.array(df[['experience', 'test_score', 'interview_score']])
y = np.array(df['salary'])

# Training the model
reg.fit(X, y)

# Predicting the output
print(reg.predict([[12, 10, 10]]))
