"""
Author @ Mihir_Srivastava
Dated - 18-05-2020
File - Predicting_class_of_breast_cancer_SVM
Aim - To predict the class of breast cancer (benign or malignant) using the breast cancer winsconsin data set by using
Support Vector Machines (SVM) algorithm.
"""

# import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
import numpy as np

# Read csv file
df = pd.read_csv('breast-cancer-wisconsin.data')

# Use the following code snippet to see data of all columns in pycharm (it actually displays '...' after 2 columns if
# there are more than 5 columns to display)
desired_width = 320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', 12)

# Filling in missing values
df.replace('?', -99999, inplace=True)

# Dropping unnecessary columns
df.drop(['id'], axis=1, inplace=True)

# Create svm object
model = svm.SVC(kernel='linear')

# Define features and labels
X = np.array(df.drop(['class'], axis=1))
y = np.array(df['class'])

# Split the data into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train the model
model.fit(X_train, y_train)

# Find accuracy
accuracy = model.score(X_test, y_test)
print("accuracy: " + str(accuracy))

# Make prediction
example = np.array([[8, 5, 2, 3, 1, 2, 7, 2, 6]])
predict = model.predict(example)

if predict == 2:
    print("benign")
else:
    print("malignant")
