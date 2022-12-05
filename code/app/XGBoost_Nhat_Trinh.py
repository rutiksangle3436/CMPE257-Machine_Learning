#!/usr/bin/env python
# coding: utf-8

# # **CMPE 257 Project Milestone 2**
# ## **Team 8**

# **Installing and importing all required libraries**

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


# **Importing data from CSV file using read_csv()**

# In[2]:


covid = pd.read_csv("WHO-COVID-19-global-data.csv")
covid.head()


# **Preprocessing the data to extract New cases and Cumulative cases for a specific country**

# In[3]:


india = covid.loc[covid['Country'] == 'India']


# In[4]:


india = india[['Date_reported', 'New_cases', 'Cumulative_cases']].reset_index()
india = india[['Date_reported', 'New_cases', 'Cumulative_cases']]
india


# **Dividing the data into training and testing data (approx 80:20)**

# In[5]:


newC = india['New_cases'][:800]
cumC = india['Cumulative_cases'][:800]

newCtest = india['New_cases'][800:]

cumCtest = india['Cumulative_cases'][800:].reset_index()
cumCtest = cumCtest['Cumulative_cases']

miss1 = india['Cumulative_cases']
miss1

df = pd.DataFrame({'date':pd.date_range('2020-01-03', periods=len(miss1)), 'cases':miss1})
df['date'] = df['date'].astype(int)


# **Dividing the data in frames and appending in a list for time series forecasting**

# In[6]:


from numpy import array

X, y = list(), list()
for i in range(0,len(cumC),1):
  # find the end of this pattern
  end_ix = i + 3
  # check if we are beyond the sequence
  if end_ix > len(cumC)-1:
    break
  # gather input and output parts of the pattern
  seq_x, seq_y = cumC[i:end_ix], cumC[end_ix]
  X.append(seq_x)
  y.append(seq_y)



X = array(X) 
y = array(y)

# for i in range(len(X)):
# 	print(X[i], y[i])
X


# In[7]:


from numpy import array

Xtest, ytest = list(), list()
for i in range(len(cumCtest)):
  # find the end of this pattern
  end_ix = i + 3
  # check if we are beyond the sequence
  if end_ix > len(cumCtest)-1:
    break
  # gather input and output parts of the pattern
  seq_x, seq_y = cumCtest[i:end_ix], cumCtest[end_ix]
  Xtest.append(seq_x)
  ytest.append(seq_y)



Xtest = array(Xtest) 
ytest = array(ytest)
actual = ytest
# for i in range(len(X)):
	# print(X[i], y[i])
 
# print(len(actual))
# actual


# # **XGBoost**

# **Creating the XGBoodt model and printing the predicted and expected values along with a plot showing the results**

# In[8]:


from numpy import asarray
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from matplotlib import pyplot
# transform a time series dataset into a supervised learning dataset
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols = list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
	# put it all together
	agg = concat(cols, axis=1)
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg.values
 
# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
	return data[:-n_test, :], data[-n_test:, :]
 
# fit an xgboost model and make a one step prediction
def xgboost_forecast(train, testX):
	# transform list into array
	train = asarray(train)
	# split into input and output columns
	trainX, trainy = train[:, :-1], train[:, -1]
	# fit model
	model = XGBRegressor(objective='reg:squarederror', n_estimators=1000)
	model.fit(trainX, trainy)
	# make a one-step prediction
	yhat = model.predict(asarray([testX]))
	return yhat[0]
 
# walk-forward validation for univariate data
def walk_forward_validation(data, n_test):
	predictions = list()
	# split dataset
	train, test = train_test_split(data, n_test)
	# seed history with training dataset
	history = [x for x in train]
	# step over each time-step in the test set
	for i in range(len(test)):
		# split test row into input and output columns
		testX, testy = test[i, :-1], test[i, -1]
		# fit model on history and make a prediction
		yhat = xgboost_forecast(history, testX)
		# store forecast in list of predictions
		predictions.append(yhat)
		# add actual observation to history for the next loop
		history.append(test[i])
		# summarize progress
		print('>expected=%.1f, predicted=%.1f' % (testy, yhat))
	# estimate prediction error
	error = mean_absolute_error(test[:, -1], predictions)
	return error, test[:, -1], predictions
 
# load the dataset
data = series_to_supervised(df, n_in=6)
print(data)
# evaluate
mae, y, yhat = walk_forward_validation(data, 230)
print('MAE: %.3f' % mae)
# plot expected vs preducted
pyplot.plot(y, label='Expected')
pyplot.plot(yhat, label='Predicted')
pyplot.legend()
pyplot.show()

