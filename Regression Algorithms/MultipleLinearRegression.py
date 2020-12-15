# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 17:23:50 2020

@author: Vandan
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
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
clf = LinearRegression(random_state=0)
clf = clf.fit(X_train,y_train)

# Predict
y_pred = clf.predict(X_test)

