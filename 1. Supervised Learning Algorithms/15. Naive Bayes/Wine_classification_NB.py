"""
Author @ Mihir_Srivastava
Dated - 23-05-2020
File - Wine_classification_NB
Aim - To predict the class of wine based on certain parameters using Naive Bayes algorithm.
"""

# import necessary libraries
import numpy as np
from sklearn import naive_bayes
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine

wine = load_wine()

# Define features and labels
X = np.array(wine.data)
y = np.array(wine.target)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = naive_bayes.GaussianNB()

# Training the model
model.fit(X_train, y_train)

# Finding accuracy by testing the model
accuracy = model.score(X_test, y_test)

print("Accuracy of our model: ", accuracy)

# print(dir(wine))


