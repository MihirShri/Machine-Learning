"""
Author @ Mihir_Srivastava
Dated - 20-05-2020
File - Iris_flower_classification_RFC
Aim - To predict the class of the iris flower by training the data set "load_iris" available in sklearn library using
the Random Forest Classifier (RFC) algorithm.
"""

# import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn

# Create load_iris object
iris = load_iris()

# Convert it into a DataFrame for better visualization
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Add some more details for better understandability
df['target'] = iris.target
df['target_names'] = df.target.apply(lambda x: iris.target_names[x])

# Create Random Forest Classifier model
model = RFC()

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

predict = model.predict(X_test)

# Creating confusion matrix
cm = confusion_matrix(y_test, predict)

# Visualizing the confusion  matrix
plt.figure(figsize=(10, 7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()
