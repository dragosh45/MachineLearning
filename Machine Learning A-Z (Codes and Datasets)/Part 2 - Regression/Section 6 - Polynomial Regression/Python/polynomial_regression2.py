# -*- coding: utf-8 -*-
"""
Created on Fri May  7 21:47:07 2021

@author: h1dr0
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# train
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

# train the poly matrx : x square
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

#visualising the linear regression results
plt.scatter(X,y ,color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title("Truthe or Bluff (linear Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

#visualising the linear poly regression results
plt.scatter(X,y ,color = 'red')
plt.plot(X, lin_reg_2.predict(X_poly), color = 'blue')
plt.title("Truthe or Bluff (linear Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

#predict for 6 with lin reg
lin_reg.predict([[6.5]])

#predict for 6 with lin poly reg
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))