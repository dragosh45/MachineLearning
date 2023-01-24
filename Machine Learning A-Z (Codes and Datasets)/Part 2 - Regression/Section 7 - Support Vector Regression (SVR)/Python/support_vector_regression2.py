# -*- coding: utf-8 -*-

#Data preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values 
print(X)
print(y)
#transform y format like X for fit_transform function takes like x format
y = y.reshape(len(y),1)
print(y)
#apply feature scalling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
" fit get the mean() and standard deviation(x) din formula"
" transform will do the forumula for standardisation "
y = sc_y.fit_transform(y)
print(X)
print(y)


#Training the model with above data

from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, y.ravel())

#predict the result
"standardisation with transform"
"reversing the feature scalling of y"

sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])))