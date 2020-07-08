"""
Author @ Mihir_Srivastava
Dated - 15-05-2020
File - Gradient_descent exercise
Aim - To compare between the Linear Regression function of sklearn library and the gradient descent function created by us.
"""

import numpy as np
import pandas as pd
import math
from sklearn.linear_model import LinearRegression


# Create a function gradient descent
def gradient_descent(x, y):
    # Define current values of m and b
    m_curr = b_curr = 0

    # Define the number of iterations
    iterations = 1000000

    # Define the value on n (total number of points)
    n = len(x)

    # Define learning_rate
    learning_rate = 0.0002

    cost_previous = 0

    for i in range(iterations):
        # y = m * x + b
        y_predicted = m_curr * x + b_curr

        # The Mean Squared Error (MSE)
        cost = (1 / n) * sum([val ** 2 for val in (y - y_predicted)])

        # Partial derivatives with respect to m and b (direction to move next)
        md = -(2 / n) * sum(x * (y - y_predicted))
        bd = -(2 / n) * sum(y - y_predicted)

        # step length (how much to move)
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd

        # stop if cost gets close to e ^ (-20)
        if math.isclose(cost, cost_previous, rel_tol=1e-20):
            break

        cost_previous = cost

    # Return m and b
    return m_curr, b_curr


def predict_using_sklearn():
    # Read csv file
    d = pd.read_csv("test_scores.csv")

    # Create Linear Regression object
    r = LinearRegression()

    # Train the model
    r.fit(d[['math']], d.cs)

    # Return m and b
    return r.coef_, r.intercept_


# Read csv file containing scores
df = pd.read_csv('test_scores.csv')

# Data points need to be passed as arrays
x = np.array(df['math'])
y = np.array(df['cs'])

# Compare m and b
print(gradient_descent(x, y))
print(predict_using_sklearn())
