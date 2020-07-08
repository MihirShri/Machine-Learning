"""
Author @ Mihir_Srivastava
Dated - 15-05-2020
File - Gradient_descent
Aim - To see how Linear Regression actually works by finding the slope (m) and intercept (b) of a line given different
values of x and y using the concept of Gradient descent
"""

import numpy as np


# Create a function gradient descent
def gradient_descent(x, y):
    # Define current values of m and b
    m_curr = b_curr = 0

    # Define the number of iterations
    iterations = 1226

    # Define the value on n (total number of points)
    n = len(x)

    # Define learning_rate
    learning_rate = 0.08

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

        print("m {}, b {}, cost {}, iteration {}".format(m_curr, b_curr, cost, i))


# Data points need to be passed as arrays
x = np.array([1, 2, 3, 4, 5])
y = np.array([5, 7, 9, 11, 13])

# Call the function
gradient_descent(x, y)
