"""
Author @ Mihir_Srivastava
Dated - 17-05-2020
File - Predicting_insurance
Aim - To predict whether a person will buy an insurance or not given his age.
"""

# import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np

# Read csv file
df = pd.read_csv('insurance_data.csv')

# Visualize the data using a scatter plot
plt.title('Insurance taken or not')
plt.xlabel('age')
plt.ylabel('bought insurance')
plt.scatter(df['age'], df['bought_insurance'], marker='+', color='r')
plt.show()

# Create Logistic Regression model
model = LogisticRegression()

# Define features and labels
X = np.array(df[['age']])
y = np.array(df['bought_insurance'])

# Split the data into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# Train the model
model.fit(X_train, y_train)

# Find accuracy
accuracy = model.score(X_test, y_test)

# Printing the probabilities
print('Probability of [buying (1), not buying (0)]: ' + str(model.predict_proba([[38]])))

# Making prediction
predict = model.predict([[38]])

if predict == 1:
    print('Will buy')
else:
    print('Will not buy')
