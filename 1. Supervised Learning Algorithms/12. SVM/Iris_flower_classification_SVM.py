"""
Author @ Mihir_Srivastava
Dated - 19-05-2020
File - Iris_flower_classification_SVM
Aim - To predict the class of the iris flower by training the data set "load_iris" available in sklearn library using
the Support Vector Machines (SVM) algorithm.
"""

# import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.datasets import load_iris

# Create load_iris object
iris = load_iris()

# Convert it into a DataFrame for better visualization
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Add some more details for better understandability
df['target'] = iris.target
df['target_names'] = df.target.apply(lambda x: iris.target_names[x])

# Create svm model
model = svm.SVC()

# Define features and labels
X = df.drop(['target', 'target_names'], axis=1)
y = iris.target

# Split the data into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train the model
model.fit(X_train, y_train)

# Find accuracy
accuracy = model.score(X_test, y_test)
print("accuracy: " + str(accuracy))
