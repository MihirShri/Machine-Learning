"""
Author @ Mihir_Srivastava
Dated - 19-05-2020
File - my_own_SVM_algo
Aim - To implement my own SVM algorithm, train it, and use it to make predictions and visualise the results.
"""

# Import necessary libraries
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

style.use('ggplot')


# Creating class for SVM (Note that we didn't create a class for Linear Regression or KNN because in SVM, we do not want
# to train the model again and again which was not the case with KNN or LR)
class Support_Vector_Machine:
    # Defining constructor
    def __init__(self, visualization=True):
        self.visualization = visualization
        self.colors = {1: 'r', -1: 'b'}
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1, 1, 1)

    # Training the model
    def fit(self, data):
        self.data = data

        # Our optimum dictionary will be of the type - { ||w||: [w,b] }
        opt_dict = {}

        # We want to check for all values of w as they all have the same magnitude with different directions.
        # (Suppose w = [5, 5] then [-5, 5], [-5, -5] and [5, -5] also have the same magnitude but different directions)
        transforms = [[1, 1],
                      [-1, 1],
                      [-1, -1],
                      [1, -1]]

        # A list which will contain all the points in our data set
        all_data = []
        for yi in self.data:
            for featureset in self.data[yi]:
                for feature in featureset:
                    all_data.append(feature)

        # Selecting max and min values out of all those points to provide a good starting coordinate for w
        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)
        all_data = None

        # These are the sizes of the steps w will take after each iteration. If you want to be more accurate, go for an
        # even lower value at the expense of a highly increased computation time. So, there is a trade of between
        # accuracy and time take as in most of the algorithms.
        step_sizes = [self.max_feature_value * 0.1,
                      self.max_feature_value * 0.01,
                      # point of expense:
                      self.max_feature_value * 0.001]

        # extremely expensive
        b_range_multiple = 5

        # This is the size of the step b will be taking. Note that, minimizing w is more important than maximizing b so
        # size of the steps of b doesn't matter much. That's why we choose a high value.
        # we don't need to take as small of steps with b as we do w
        b_multiple = 5

        # Suppose max obtained is 8 so, starting point for w will be [80, 80]
        latest_optimum = self.max_feature_value * 10

        # For each step, do the following
        for step in step_sizes:
            w = np.array([latest_optimum, latest_optimum])

            # we can do this because it is a convex optimization problem
            optimized = False

            while not optimized:
                # Defining the starting, ending and the size of each step for b.
                for b in np.arange(-1 * (self.max_feature_value * b_range_multiple),
                                   self.max_feature_value * b_range_multiple,
                                   step * b_multiple):

                    # Checking for all values of w by multiplying it with the transforms list we created
                    for transformation in transforms:
                        w_t = w * transformation
                        found_option = True
                        # weakest link in the SVM fundamentally, SMO attempts to fix this a bit
                        # checking if yi(xi.w+b) >= 1
                        for i in self.data:
                            for xi in self.data[i]:
                                yi = i
                                if not yi * (np.dot(w_t, xi) + b) >= 1:
                                    found_option = False

                        if found_option:
                            # Populating our optimum dictionary
                            opt_dict[np.linalg.norm(w_t)] = [w_t, b]

                # Since we've already checking for all values of w by multiplying it with transforms, we do not want
                # to go any further
                if w[0] < 0:
                    optimized = True
                    print('Optimized a step.')
                else:
                    w = w - step

            norms = sorted([n for n in opt_dict])
            # Note that, our dictionary is of the form - { ||w|| : [w,b] }
            opt_choice = opt_dict[norms[0]]
            self.w = opt_choice[0]
            self.b = opt_choice[1]
            latest_optimum = opt_choice[0][0] + step * 2

        # An unnecessary step. Doing this just to print the values of different points just to check how close they are
        # to their hyperplanes.
        for i in self.data:
            for xi in self.data[i]:
                yi = i
                print(xi, ':', yi * (np.dot(self.w, xi) + self.b))

    # Prediction function
    def predict(self, features):
        # return sign( x.w + b ) --> (Equation of the hyperplane)
        classification = np.sign(np.dot(np.array(features), self.w) + self.b)
        if classification != 0 and self.visualization:
            self.ax.scatter(features[0], features[1], s=50, marker='*', c=self.colors[classification])
        return classification


    def visualize(self):
        [[self.ax.scatter(x[0], x[1], s=100, color=self.colors[i]) for x in data_dict[i]] for i in data_dict]

        # hyperplane = x.w + b
        # v = x.w + b

        # Values of v for different cases are:
        # psv = 1 (Positive Support Vector)
        # nsv = -1 (Negative Support Vector)
        # db = 0 (Decision boundary)

        def hyperplane(x, w, b, v):
            return (-w[0] * x - b + v) / w[1]

        # Defining the range of our graph
        datarange = (self.min_feature_value * 0.9, self.max_feature_value * 1.1)
        hyp_x_min = datarange[0]
        hyp_x_max = datarange[1]

        # (w.x + b) = 1
        # positive support vector hyperplane
        psv1 = hyperplane(hyp_x_min, self.w, self.b, 1)
        psv2 = hyperplane(hyp_x_max, self.w, self.b, 1)
        self.ax.plot([hyp_x_min, hyp_x_max], [psv1, psv2], 'k')

        # (w.x + b) = -1
        # negative support vector hyperplane
        nsv1 = hyperplane(hyp_x_min, self.w, self.b, -1)
        nsv2 = hyperplane(hyp_x_max, self.w, self.b, -1)
        self.ax.plot([hyp_x_min, hyp_x_max], [nsv1, nsv2], 'k')

        # (w.x + b) = 0
        # decision boundary
        db1 = hyperplane(hyp_x_min, self.w, self.b, 0)
        db2 = hyperplane(hyp_x_max, self.w, self.b, 0)
        self.ax.plot([hyp_x_min, hyp_x_max], [db1, db2], 'y--')

        plt.show()


# Our data set
data_dict = {-1: np.array([[1, 7],
                           [2, 8],
                           [3, 8], ]),

             1: np.array([[5, 1],
                          [6, -1],
                          [7, 3], ])}

# Creating SVM object
svm = Support_Vector_Machine()

# Training the model
svm.fit(data=data_dict)

# Points for which we want to predict the class
predict_us = [[0, 10],
              [1, 3],
              [3, 4],
              [3, 5],
              [5, 5],
              [5, 6],
              [6, -5],
              [5, 8],
              [3, 3.5]]

# Making prediction
for p in predict_us:
    svm.predict(p)

# Visualizing everything
svm.visualize()
