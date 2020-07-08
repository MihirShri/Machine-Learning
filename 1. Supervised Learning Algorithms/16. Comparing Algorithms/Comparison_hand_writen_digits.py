"""
Author @ Mihir_Srivastava
Dated - 20-05-2020
File - Comparison_Hand_written_digits_recognition
Aim - To compare the accuracy of different algorithms (Decision tree, K-nearest-neighbors [KNN], Support VectorMachines
[SVM], Random Forest [RF]) by testing against the load_digits DataSet of sklearn library.
"""

# import necessary libraries
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_digits
from collections import OrderedDict

# Create load_digits object
digits = load_digits()

# Define features and labels
Xs = digits.data
ys = digits.target

# Use cross_val_score to train and test all batches
scores_dt = cross_val_score(DecisionTreeClassifier(), Xs, ys, cv=10)
scores_knn = cross_val_score(KNeighborsClassifier(), Xs, ys, cv=10)
scores_svm = cross_val_score(SVC(kernel='poly'), Xs, ys, cv=10)
scores_rf = cross_val_score(RandomForestClassifier(n_estimators=50), Xs, ys, cv=10)

print("Decision Tree:", sum(scores_dt) / len(scores_dt))
print("KNN:", sum(scores_knn) / len(scores_knn))
print("SVM:", sum(scores_svm) / len(scores_svm))
print("Random Forest:", sum(scores_rf) / len(scores_rf))

result = {"Decision Tree": sum(scores_dt) / len(scores_dt), "KNN": sum(scores_knn) / len(scores_knn),
          "SVM": sum(scores_svm) / len(scores_svm), "Random Forest": sum(scores_rf) / len(scores_rf)}

print("\n")
print("Algorithms in sorted order of their accuracy: ")

dic = OrderedDict(sorted(result.items(), key=lambda x: x[1]))
print(dic)
