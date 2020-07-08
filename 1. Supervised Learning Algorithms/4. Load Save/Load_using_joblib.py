"""
Author @ Mihir_Srivastava
Dated - 16-05-2020
File - Load_using_pickle
Aim - To load an already trained Linear Regression model saved using joblib and use it to predict the output
"""

# Import the required libraries
import joblib

# Load the saved model in a variable
mj = joblib.load('model_joblib')

# Predict using that variable
print(mj.predict([[3300]]))
