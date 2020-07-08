"""
Author @ Mihir_Srivastava
Dated - 17-05-2020
File - Iris_flower_classification_Logistic
Aim - To predict the class of the iris flower by training the data set "load_iris" available in sklearn library using
Logistic Regression algorithm.
"""

# import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn

# Create load_iris object
iris = load_iris()

# Create Logistic Regression model
model = LogisticRegression(max_iter=7600)

# Define features and labels
X = iris.data
y = iris.target

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

# Making a prediction
prediction = model.predict([[5.9, 3.0, 5.1, 1.8]])

if prediction == 0:
    print("setosa")
elif prediction == 1:
    print("versicolor")
else:
    print("virginica")
