"""
Author @ Mihir_Srivastava
Dated - 20-05-2020
File - Comparison_employee_retention
Aim - To compare the accuracy of different algorithms (Logistic Regression, Decision tree, K-nearest-neighbors [KNN]
Random Forest [RF]) by testing against the employee retention DataSet.
"""

# import necessary libraries
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from collections import OrderedDict

# Read csv file
df = pd.read_csv('HR_comma_sep.csv')

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

# Define features and labels
X = new_df
y = df.left

# Use cross_val_score to train and test all batches
scores_lr = cross_val_score(LogisticRegression(max_iter=7600), X, y, cv=10)
scores_dt = cross_val_score(DecisionTreeClassifier(), X, y, cv=10)
scores_knn = cross_val_score(KNeighborsClassifier(), X, y, cv=10)
scores_rf = cross_val_score(RandomForestClassifier(n_estimators=40), X, y, cv=10)

print("Logistic Regression:", sum(scores_lr) / len(scores_lr))
print("Decision Tree:", sum(scores_dt) / len(scores_dt))
print("KNN:", sum(scores_knn) / len(scores_knn))
print("Random Forest:", sum(scores_rf) / len(scores_rf))

result = {"Logistic Regression": sum(scores_lr) / len(scores_lr), "Decision Tree": sum(scores_dt) / len(scores_dt),
          "KNN": sum(scores_knn) / len(scores_knn), "Random Forest": sum(scores_rf) / len(scores_rf)}

print("\n")
print("Algorithms in sorted order of their accuracy: ")

dic = OrderedDict(sorted(result.items(), key=lambda x: x[1]))
print(dic)
