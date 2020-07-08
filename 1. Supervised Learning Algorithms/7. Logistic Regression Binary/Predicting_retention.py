"""
Author @ Mihir_Srivastava
Dated - 17-05-2020
File - Predicting_retention
Aim - To predict whether a person will leave a company or not based on several factors.
"""

# import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np

# Read csv file
df = pd.read_csv('HR_comma_sep.csv')

# Use the following code snippet to see data of all columns in pycharm (it actually displays '...' after 2 columns if
# there are more than 5 columns to display)
desired_width = 320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', 10)

# Keeping record for the number of employees left and retained
left = df[df['left'] == 1]
retained = df[df['left'] == 0]

# Grouping the complete data in terms of value of left and calculating mean for all data
df1 = df.groupby('left').mean()

# print(df1)

# From this data we can draw following conclusions:
# 1.**Satisfaction Level**: Satisfaction level seems to be relatively low (0.44) in employees leaving the firm vs the retained ones (0.66)
# 2.**Average Monthly Hours**: Average monthly hours are higher in employees leaving the firm (199 vs 207)
# 3.**Promotion Last 5 Years**: Employees who are given promotion are likely to be retained at firm

# Impact of salary
pd.crosstab(df.salary, df.left).plot(kind='bar')
# plt.show()

# Above bar chart shows employees with high salaries are likely to not leave the company

# Impact of department
pd.crosstab(df.Department, df.left).plot(kind='bar')
# plt.show()

# Department doesn't seem to have a major impact on the retention rate. Hence, ignoring.

# From our data analysis, we conclude that we will use the following columns as our features
# 1.**Satisfaction Level**
# 2.**Average Monthly Hours**
# 3.**Promotion Last 5 Years**
# 4.**Salary**

new_df = df[['satisfaction_level', 'average_montly_hours', 'promotion_last_5years', 'salary']]

# Handling the salary column with dummy variable technique

# Get dummy variables (columns)
d = pd.get_dummies(df['salary'])

# Concatenate the dummy columns to original dataframe
merged = pd.concat([new_df, d], axis=1)

# Drop the unnecessary columns
new_df = merged.drop(['salary'], axis=1)

# Create Logistic Regression model
model = LogisticRegression()

# Define features and labels
X = new_df
y = df.left

# Split the data into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train the model
model.fit(X_train, y_train)

# Find accuracy
accuracy = model.score(X_test, y_test)
print("accuracy: " + str(accuracy))

# Making prediction
predict = model.predict([[0.30, 250, 0, 0, 1, 0]])

if predict == 1:
    print("will leave")
else:
    print("will not leave")
