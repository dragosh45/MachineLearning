# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 14:19:05 2021

@author: h1dr0
"""

# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)




# training the simple linear regression model on the training set

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#predict according y to x_test
y_pred = regressor.predict(X_test)

#y_train sunt punctele la valorile reale
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
#regressor.predict(X_train) sunt punctele prezise care fac o linie 'prezisa'
#cu punctele de pe x (x_train)
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.xlabel('Salary')
plt.show()

#pune punctele de test pe linia prezisa
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.xlabel('Salary')
plt.show()