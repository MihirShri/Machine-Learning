"""
Author @ Mihir_Srivastava
Dated - 17-05-2020
File - Hand_written_digits_recognition
Aim - To predict the hand written digits by training the dataset "load_digits" available in sklearn library using the
Logistic Regression algorithm.
"""

# import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn

# Create load_digits object
digits = load_digits()

# Create Logistic Regression model
model = LogisticRegression(max_iter=7600)

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

# Predicting the test set
predict = model.predict(X_test)

# Preparing a confusion matrix
cm = confusion_matrix(y_test, predict)

# Visualizing the confusion  matrix
plt.figure(figsize=(10, 7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()
