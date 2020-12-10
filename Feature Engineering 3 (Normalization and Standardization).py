# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 17:23:53 2020

@author: Vandan
"""
################################################################################################

# Standardization and Normalization

################################################################################################

#Transformation of Features
#Why Transformation of Features Are Required?
#
# Linear Regression---Gradient Descent ----Global Minima
# Helps to calculate gradients efficiently


#Algorithms like KNN,K Means,Hierarichal Clustering--- Eucledian Distance
#Every Point has some vectors and Directiom

# It helps when one variable is ranging in small scale and other variable is ranging in large scale

#Deep Learning Techniques(Standardization, Scaling) 1.ANN--->GLobal Minima, Gradient 2.CNN 3.RNN
#
#0-255 pixels
#


#Types Of Transformation
#1. Normalization And Standardization
#2. Scaling to Minimum And Maximum values
#3. Scaling To Median And Quantiles
#4.  Guassian Transformation 
#    Logarithmic Transformation 
#    Reciprocal Trnasformation 
#    Square Root Transformation
#    Exponential Trnasformation 
#    Box Cox Transformation



# 1.  Standardization
# We try to bring all the variables or features to a similar scale. standarisation
# means centering the variable at zero. z=(x-x_mean)/std

import pandas as pd
df=pd.read_csv('titanic.csv', usecols=['Pclass','Age','Fare','Survived'])
df.head()
df['Age'].fillna(df.Age.median(),inplace=True)


#### standarisation: We use the Standardscaler from sklearn library
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
### fit vs fit_transform
df_scaled=scaler.fit_transform(df)


import matplotlib.pyplot as plt
%matplotlib inline
plt.hist(df_scaled[:,2],bins=20)

# works well with most of all algortithms
# If you have outliers it may impact results 






# 2. Normalization (min-max)
# Min Max Scaling (### CNN)--- works well in Deep Learning Techniques
# Min Max Scaling scales the values between 0 to 1. X_scaled = (X - X.min / (X.max - X.min)
from sklearn.preprocessing import MinMaxScaler
min_max=MinMaxScaler()
df_minmax=pd.DataFrame(min_max.fit_transform(df),columns=df.columns)
df_minmax.head()
plt.hist(df_minmax['Age'],bins=20)





# 3. Robust scalar
#It is used to scale the feature to median and quantiles Scaling using median and quantiles consists of substracting the median to all the observations, and then dividing by the interquantile difference. The interquantile difference is the difference between the 75th and 25th quantile:
#
#IQR = 75th quantile - 25th quantile
#
#X_scaled = (X - X.median) / IQR
#
#0,1,2,3,4,5,6,7,8,9,10
#
#9-90 percentile---90% of all values in this group is less than 9 1-10 precentile---10% of all values in this group is less than 1 4-40%

from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
df_robust_scaler=pd.DataFrame(scaler.fit_transform(df),columns=df.columns)
df_robust_scaler.head()
plt.hist(df_minmax['Age'],bins=20)
# It is robust to outliers




# 4. Gaussian Distribution
Guassian Transformation
# Some machine learning algorithms like linear and logistic
# assume that the features are normally distributed -Accuracy -Performance
df=pd.read_csv('titanic.csv', usecols=['Pclass','Age','Fare','Survived'])
df.head()
df['Age'].fillna(df.Age.median(),inplace=True)

import scipy.stats as stat
import pylab

# If you want to check whether feature is guassian or normal distributed
# Q-Q plot
def plot_data(df,feature):
    plt.figure(figsize=(10,6))
    plt.subplot(1,2,1)
    df[feature].hist()
    plt.subplot(1,2,2)
    stat.probplot(df[feature],dist='norm',plot=pylab)
    plt.show()

plot_data(df,'Age')



# Logarithmic distribution 
import numpy as np
df['Age_log']=np.log(df['Age'])
plot_data(df,'Age_log')


# Reciprocal Trnasformation 
df['Age_reciprocal']=1/df.Age
plot_data(df,'Age_reciprocal')



# Square Root Transformation
df['Age_sqaure']=df.Age**(1/2)
plot_data(df,'Age_sqaure')



# Exponential Trnasformation 
df['Age_exponential']=df.Age**(1/1.2)
plot_data(df,'Age_exponential')



# Box Cox Transformation
df['Age_Boxcox'],parameters=stat.boxcox(df['Age'])
plot_data(df,'Age_Boxcox')







