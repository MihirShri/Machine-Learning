"""
Author @ Mihir_Srivastava
Dated - 23-05-2020
File - Detecting_spam_mails_NB
Aim - To detect whether an E-mail is spam or not using Naive Bayes algorithm.
"""

# import necessary libraries
import pandas as pd
import numpy as np
from sklearn import naive_bayes
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

# Read csv file
df = pd.read_csv('spam.csv')

# print(df.groupby('Category').describe())

# categorizing spam and ham emails as 1 and 0 respectively.
df['spam'] = df['Category'].apply(lambda x: 1 if x == 'spam' else 0)

# using Pipeline to eliminate the need of transforming from Vectorizer each time.
model = Pipeline([('vectorizer', CountVectorizer()),
                  ('nb', naive_bayes.MultinomialNB())])

# Define features and labels
X = np.array(df.Message)
y = np.array(df.spam)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Training the model
model.fit(X_train, y_train)

# Finding accuracy by testing the model
accuracy = model.score(X_test, y_test)

emails = []
e = input("Enter the mail: ")
emails.append(e)

# Making prediction
y = model.predict(emails)

print("Accuracy of our model: ", accuracy)

if y == 0:
    print("This email looks like a Ham (Not Spam)")
else:
    print("This E-mail looks like a Spam")
