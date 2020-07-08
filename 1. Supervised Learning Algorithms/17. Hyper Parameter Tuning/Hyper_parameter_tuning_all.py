"""
Author @ Mihir_Srivastava
Dated - 23-05-2020
File - Hyper_parameter_tuning_all
Aim - To do hyper parameter tuning and figure out the best algorithm with the best parameters which give the highest
accuracy on the iris DataSet.
"""

# import necessary libraries
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
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

# Defining the models and parameters to test against
model_params = {'svm': {'model': svm.SVC(gamma='auto'), 'params': {'kernel': ['rbf', 'linear', 'poly'], 'C': [1, 5, 10, 20]}},
                'logistic_regression': {'model': LogisticRegression(solver='liblinear', multi_class='auto', max_iter=7600), 'params': {'C': [1, 5, 10, 20]}},
                'KNN': {'model': KNeighborsClassifier(), 'params': {'n_neighbors': [5, 7, 9, 11]}},
                'decision_tree': {'model': DecisionTreeClassifier(), 'params': {'splitter': ['best', 'random']}},
                'random_forest': {'model': RandomForestClassifier(), 'params': {'n_estimators': [1, 5, 10, 20, 40, 50, 60]}}}

# Define features and labels
X = df.drop(['target', 'target_names'], axis=1)
y = iris.target

scores = []
for model_name, mp in model_params.items():
    model = GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False)
    # Train the model
    model.fit(X, y)
    scores.append({'model': model_name,
                   'best_score': model.best_score_,
                   'best_parameters': model.best_params_})

df = pd.DataFrame(scores, columns=['model', 'best_score', 'best_parameters'])

print(df)
