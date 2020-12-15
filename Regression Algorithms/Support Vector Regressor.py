# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 17:54:26 2020

@author: Vandan
"""

import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

Data = pd.read_csv("winequality-red.csv")
print(Data.head(5))


# Spliting features
# Features
X = Data.iloc[:,0:11]
# Target variable
y = Data.quality

# Dividing train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)

# Building a model
reg = SVR()
reg = reg.fit(X_train,y_train)

# Predict
y_pred = reg.predict(X_test)


# Explained variance
from sklearn.metrics import explained_variance_score
print('Explained Variance score: ',explained_variance_score(y_test, y_pred))


from sklearn.metrics import mean_squared_error
print('Mean Squared Error: ',mean_squared_error(y_test, y_pred))

# Mean Absolute error
from sklearn.metrics import mean_absolute_error
print('Mean Absolute Error: ',mean_absolute_error(y_test, y_pred))

# Max error
from sklearn.metrics import max_error
print('Maximum error: ',max_error(y_test, y_pred))

# R Squared
from sklearn.metrics import r2_score
print('R-Squared: ',r2_score(y_test, y_pred))

# Adjusted R Square
import numpy as np
y_pred = reg.predict(X_test)
SS_Residual = sum((y_test-y_pred)**2)       
SS_Total = sum((y_test-np.mean(y))**2)     
r_squared = 1 - (float(SS_Residual))/SS_Total
adjusted_r_squared = 1 - (1-r_squared)*(len(y_test)-1)/(len(y_test)-X_train.shape[1]-1)
print('Adjusted R Square: ',adjusted_r_squared)
