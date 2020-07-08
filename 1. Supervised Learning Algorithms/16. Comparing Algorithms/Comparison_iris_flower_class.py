"""
Author @ Mihir_Srivastava
Dated - 20-05-2020
File - Comparison_iris_flower_classification
Aim - To compare the accuracy of different algorithms (Logistic Regression, Decision tree, K-nearest-neighbors [KNN],
Support Vector Machines [SVM], Random Forest [RF]) by testing against the load_iris DataSet of sklearn library.
"""

# import necessary libraries
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from collections import OrderedDict

# Create load_iris object
iris = load_iris()

# Define features and labels
X = iris.data
y = iris.target

# Use cross_val_score to train and test all batches
scores_lr = cross_val_score(LogisticRegression(max_iter=7600), X, y, cv=10)
scores_dt = cross_val_score(DecisionTreeClassifier(), X, y, cv=10)
scores_knn = cross_val_score(KNeighborsClassifier(), X, y, cv=10)
scores_svm = cross_val_score(SVC(kernel='linear'), X, y, cv=10)
scores_rf = cross_val_score(RandomForestClassifier(n_estimators=50), X, y, cv=10)

print("Logictic Regression:", sum(scores_lr) / len(scores_lr))
print("Decision Tree:", sum(scores_dt) / len(scores_dt))
print("KNN:", sum(scores_knn) / len(scores_knn))
print("SVM:", sum(scores_svm) / len(scores_svm))
print("Random Forest:", sum(scores_rf) / len(scores_rf))

result = {"Logistic Regression": sum(scores_lr) / len(scores_lr), "Decision Tree": sum(scores_dt) / len(scores_dt),
          "KNN": sum(scores_knn) / len(scores_knn), "SVM": sum(scores_svm) / len(scores_svm),
          "Random Forest": sum(scores_rf) / len(scores_rf)}

print("\n")
print("Algorithms in sorted order of their accuracy: ")

dic = OrderedDict(sorted(result.items(), key=lambda x: x[1]))
print(dic)
