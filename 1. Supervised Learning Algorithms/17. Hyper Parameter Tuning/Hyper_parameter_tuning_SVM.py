"""
Author @ Mihir_Srivastava
Dated - 23-05-2020
File - Hyper_parameter_tuning_SVM
Aim - To do hyper parameter tuning on SVM algorithm to find the best parameters which give the highest accuracy on the
iris DataSet.
"""

# import necessary libraries
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV

# Create load_iris object
iris = load_iris()

# Use the following code snippet to see data of all columns in pycharm (it actually displays '...' after 2 columns if
# there are more than 5 columns to display)
desired_width = 320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', 15)

# Convert it into a DataFrame for better visualization
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Add some more details for better understandability
df['target'] = iris.target
df['target_names'] = df.target.apply(lambda x: iris.target_names[x])

# Create svm model
model = GridSearchCV(svm.SVC(gamma='auto'), {'C': [1, 10, 20], 'kernel': ['rbf', 'linear']}, cv=5, return_train_score=False)

# Define features and labels
X = df.drop(['target', 'target_names'], axis=1)
y = iris.target

# Train the model
model.fit(X, y)

# Create a DataFrame of the results
df = pd.DataFrame(model.cv_results_)

# Display only those columns that are useful
print(df[['param_C', 'param_kernel', 'params', 'mean_test_score', 'rank_test_score']])
print("Best score:", model.best_score_)
print("Best parameters:", model.best_params_)
