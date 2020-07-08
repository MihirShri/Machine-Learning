"""
Author @ Mihir_Srivastava
Dated - 19-05-2020
File - Hand_written_digits_recognition_SVM
Aim - To predict the hand written digits by training the dataset "load_digits" available in sklearn library using the
Support Vector Machines (SVM) algorithm.
"""

# import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.datasets import load_digits

# Create load_digits object
digits = load_digits()

# Create Logistic Regression model
model = svm.SVC(kernel='rbf')

# Define features and labels
X = digits.data
y = digits.target

# Split the data into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train the model
model.fit(X_train, y_train)

# Find accuracy
accuracy = model.score(X_test, y_test)
print("accuracy: " + str(accuracy))
