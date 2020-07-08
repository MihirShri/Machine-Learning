"""
Author @ Mihir_Srivastava
Dated - 16-05-2020
File - Load_using_pickle
Aim - To load an already trained Linear Regression model saved using pickle and use it to predict the output
"""

# Import the required libraries
import pickle

# Load the saved model in a variable
with open('model_pickle', 'rb') as f:
    mp = pickle.load(f)

# Predict using that variable
print(mp.predict([[3300]]))
