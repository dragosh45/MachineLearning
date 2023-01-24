# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 15:30:43 2021

@author: h1dr0
"""

#import library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset, feature variable x and dependent variable y
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
""" : takes all rows 0:3 takes 0,1,3 rows """
""" :-1 from 0 to penultima"""
""" -1 este ultima coloana"""

#prints
print(x)
print(y)

#taking care of missing data

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])
""" create instance of class SimpleImputer """
""" fit specifica unde se va pune"""
" transform ce sa transforme"
" strategy mean stie deja sa faca average la datele din toata coloana"
print(x)

#encoding the independent variable
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])] , remainder='passthrough')
x = np.array(ct.fit_transform(x)) # conver to np array what we expect for the machinery model
"in notes 2 scrie ce am facut aici"

print(x)

#encoding the dependent variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

print(y)


#Splitting the dataset into the Training set and Test set
" train set and test set for independent variables and dependent variables"
" format expected by the feature machinery models "

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size = 0.2, random_state = 1)
" 0.2 means 20% of features( values of variables of the matrix) will go into the test set"
" random_state - will split random"

print(X_train) #matrix with 8 rows (8 observations)
print(X_test)
print(Y_train) # 8 purchased decision
print(Y_test)

#Feature Scaling

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:,3:] = sc.fit_transform(X_train[:, 3:])
" fit get the mean() and standard deviation(x) din formula"
" transform will do the forumula for standardisation "
X_test[:,3:] = sc.transform(X_test[:, 3:])
" apply only transform method on the same scalar calculated before with fit and transform"
" this is because we only took some of the rows for the specific train set"

print(X_train)
print(X_test)

