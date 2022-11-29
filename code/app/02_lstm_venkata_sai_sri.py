# -*- coding: utf-8 -*-
"""02_LSTM_Venkata_Sai_Sri.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1N-GBWZlKO7Qj0NzOLRRp9mBvhXfTGayG

# **CMPE 257 Project Milestone 2**
## **Team 8**

**Installing and importing all required libraries**
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

"""**Importing data from CSV file using read_csv()**"""

covid = pd.read_csv("WHO-COVID-19-global-data.csv")
covid.head()

"""**Preprocessing the data to extract New cases and Cumulative cases for a specific country**"""

india = covid.loc[covid['Country'] == 'India']

india = india[['Date_reported', 'New_cases', 'Cumulative_cases']].reset_index()
india = india[['Date_reported', 'New_cases', 'Cumulative_cases']]
india

"""**Dividing the data into training and testing data (approx 80:20)**"""

newC = india['New_cases'][:800]
cumC = india['Cumulative_cases'][:800]

newCtest = india['New_cases'][800:]

cumCtest = india['Cumulative_cases'][800:].reset_index()
cumCtest = cumCtest['Cumulative_cases']

miss1 = india['Cumulative_cases']
miss1

df = pd.DataFrame({'date':pd.date_range('2020-01-03', periods=len(miss1)), 'cases':miss1})
df['date'] = df['date'].astype(int)

"""**Dividing the data in frames and appending in a list for time series forecasting**"""

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

"""# **LSTM**

**Creating the LSTM Model to test the accuracy with the training and test data**
"""

from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

model = Sequential()
model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(3, 1)))
model.add(LSTM(50, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

model.fit(X, y, epochs=1000, verbose = 1)
pred = []
for i in range(len(cumCtest)):
  # find the end of this pattern
  end_ix = i + 3
  # check if we are beyond the sequence
  if end_ix > len(cumCtest)-1:
    break
  # gather input and output parts of the pattern
  seq_x= cumCtest[i:end_ix]
  x_input = array(seq_x)
  x_input = x_input.reshape((1, 3, 1))
  yhat = model.predict(x_input, verbose=0)

  pred.append(yhat[0][0])
  # print(yhat)

pred = array(pred)
# pred

"""**Plotting the Actual and Predicted data to visualize the model output**"""

import matplotlib.pyplot as plt
  
# create data
#All dates were not fitting on the X-axis, so making a list of 227 numbers
x = []
print(len(pred), len(actual))
for i in range(227):
  x.append(i)
  
# plot lines
plt.figure(figsize=(10, 10))

plt.plot(x, actual, label = "Actual", linewidth=2)
plt.plot(x, pred, label = "Prediction", linewidth=2)

plt.legend()
plt.show()