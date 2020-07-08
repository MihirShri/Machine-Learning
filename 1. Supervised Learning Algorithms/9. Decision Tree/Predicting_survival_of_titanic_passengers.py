"""
Author @ Mihir_Srivastava
Dated - 17-05-2020
File - Predicting_survival_of_titanic_passengers
Aim - To predict whether a passenger on titanic 2 will survive or not based on the existing data.
"""

# import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
import math

# Read csv file
df = pd.read_csv('titanic.csv')

# Doing some data cleaning
df.set_index('PassengerId', drop=True, inplace=True)
df = df[['Survived', 'Pclass', 'Sex', 'Age', 'Fare']]  # Dropping unnecessary columns

# Fill NaN values with mean
df['Age'].fillna(math.floor(df.Age.mean()), inplace=True)

# Create LabelEncoder object
le = LabelEncoder()

# Perform label encoding
df['Sex_n'] = le.fit_transform(df['Sex'])

# Printing just to get an idea of which labels to pass for prediction
print(df)

# Dropping unnecessary columns
df = df.drop(['Sex'], axis=1)

# Create decision tree object
model = tree.DecisionTreeClassifier()

# Define features and labels
X = df.drop(['Survived'], axis=1)
y = df.Survived

# Split the data into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train the model
model.fit(X_train, y_train)

# Find accuracy
accuracy = model.score(X_test, y_test)
print("accuracy: " + str(accuracy))

# Make prediction
predict = model.predict([[3, 0, 35, 53]])

if predict == 1:
    print("Survived")
else:
    print("Not survived")
