"""
Author @ Mihir_Srivastava
Dated - 16-05-2020
File - Best_fit_line
Aim - To see how Linear Regression actually works by finding the slope (m) and intercept (b) of the best fit line given
different values of x and y using the formulas for m and b
"""

# import necessary libraries
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

# Data points need to be passed as arrays
xs = np.array([1, 2, 3, 4, 5], dtype=np.float64)
ys = np.array([7, 9, 5, 3, 1], dtype=np.float64)


# Create a function to find the slope and intercept of the best fit line
def best_fit_slope_and_intercept(xs, ys):
    # Applying formula for m and b
    m = (((mean(xs) * mean(ys)) - mean(xs * ys)) /
         ((mean(xs) * mean(xs)) - mean(xs * xs)))
    b = mean(ys) - m * mean(xs)
    return m, b


# Call the function
m, b = best_fit_slope_and_intercept(xs, ys)
print("m = " + str(round(m, 3)))
print("b = " + str(round(b, 3)))

# Define the regression line equation
regression_line = [(m * x) + b for x in xs]

# Plotting the data points and the regression line
plt.scatter(xs, ys, color='r', label='data')
plt.plot(xs, regression_line, label='regression line')
plt.legend(loc=4)
plt.show()
