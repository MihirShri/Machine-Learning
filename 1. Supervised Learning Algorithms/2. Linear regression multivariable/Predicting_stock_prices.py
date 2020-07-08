"""
Author @ Mihir_Srivastava
Dated - 16-05-2020
File - Predicting_stock_prices
Aim - To predict the stock prices 35 days into the future using Linear Regression and plotting the predicted prices on
a graph along with the already known prices.
"""

# Import necessary libraries
import datetime
import quandl as qd
import numpy as np
import math
from sklearn import preprocessing, model_selection
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.metrics import r2_score

# Use this style in the graph
style.use('ggplot')

# using quandle to read stock prices data from 'WIKI/GOOGL'
qd.ApiConfig.api_key = "xaps6uyGeCb9bzajyuae"
df = qd.get('WIKI/GOOGL')

# Keeping necessary features to define relationships
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

# Defining relationships
df['HL_pct'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Low'] * 100.0
df['pct_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

# Keeping only necessary features
df = df[['Adj. Close', 'HL_pct', 'pct_change', 'Adj. Volume']]

# Label (that is to be predicted)
forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)

# Define how many days into the future we want to predict (here, 0.01 * len(df) = 35 days)
forecast_out = int(math.ceil(0.01 * len(df)))

# The actual label (Prices 35 days into the future)
df['label'] = df[forecast_col].shift(-forecast_out)

# Defining features and labels
X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)

# X_lately contains the dates on which we need to predict the stock prices
X_lately = X[-forecast_out:]

X = X[:-forecast_out:]
df.dropna(inplace=True)
y = np.array(df['label'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

# Create Linear Regression object (n_jobs = -1 means use all the threads)
clf = LinearRegression(n_jobs=-1)

# Train the model
clf.fit(X_train, y_train)

# Find accuracy of the model
accuracy = clf.score(X_test, y_test)

# Predict the prices
forecast_set = clf.predict(X_lately)

# Info to be fed later
df['Forecast'] = np.nan

# Note that here we hard code to get the dates of the future on which we need to predict the stock prices on the x axis
# of the graph along with the other dates.
# We need to do this because 'Date' wasn't included in our features

# Getting the last available date of our data for which the stock prices are already available
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()

# Number of seconds in a day
one_day = 86400
next_unix = last_unix + one_day

# This for loop actually appends the dates ('Date' is the index) for which we need to predict the stock prices with the
# dates for which the stock prices are already available.
# All the other feature columns are filled with NaN values except the 'Forecast' column which contains the list of
# predicted stock prices
for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += 86400
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]

# Plotting the Date vs Stock Price graph
df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.title('Predicting Stock Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

print('Stock prices for the next ' + str(forecast_out) + ' days:')
print(df['Forecast'].tail(35))
print('accuracy = ' + str(accuracy))
