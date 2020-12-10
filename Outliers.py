# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 16:03:34 2020

@author: Vandan
"""

#Which Machine LEarning Models Are Sensitive To Outliers?
#Naivye Bayes Classifier--- Not Sensitive To Outliers because it works based on probablity
#SVM-------- Not Sensitive To Outliers 
#Linear Regression---------- Sensitive To Outliers
#Logistic Regression------- Sensitive To Outliers
#Decision Tree Regressor or Classifier---- Not Sensitive
#Ensemble(RF,XGboost,GB)------- Not Sensitive
#KNN--------------------------- Not Sensitive
#Kmeans------------------------ Sensitive
#Hierarichal------------------- Sensitive
#PCA-------------------------- Sensitive
#Neural Networks-------------- Sensitive


import pandas as pd
df=pd.read_csv('titanic.csv', usecols=['Pclass','Age','Fare','Survived'])
df.head()

import seaborn as sns
sns.distplot(df['Age'].dropna())

# After inserting outliers
sns.distplot(df['Age'].fillna(100))



# Gaussian distributed
figure=df.Age.hist(bins=50)
figure.set_title('Age')
figure.set_xlabel('Age')
figure.set_ylabel('No of passenger')


# Box plot
figure=df.boxplot(column="Age")


df['Age'].describe()


# If The Data Is Normally Distributed We use this

##### Assuming Age follows A Gaussian Distribution we will calculate the boundaries which differentiates the outliers
uppper_boundary=df['Age'].mean() + 3* df['Age'].std()
lower_boundary=df['Age'].mean() - 3* df['Age'].std()
print(lower_boundary), print(uppper_boundary),print(df['Age'].mean())







#If Features Are Skewed We Use the below Technique

figure=df.Fare.hist(bins=50)
figure.set_title('Fare')
figure.set_xlabel('Fare')
figure.set_ylabel('No of passenger')

df.boxplot(column="Fare")

df['Fare'].describe()


#### Lets compute the Interquantile range to calculate the boundaries
IQR=df.Fare.quantile(0.75)-df.Fare.quantile(0.25)

lower_bridge=df['Fare'].quantile(0.25)-(IQR*1.5)
upper_bridge=df['Fare'].quantile(0.75)+(IQR*1.5)
print(lower_bridge), print(upper_bridge)



#### Extreme outliers
lower_bridge=df['Fare'].quantile(0.25)-(IQR*3)
upper_bridge=df['Fare'].quantile(0.75)+(IQR*3)
print(lower_bridge), print(upper_bridge)



# Handling outliers
data=df.copy()
data.loc[data['Age']>=73,'Age']=73

data.loc[data['Fare']>=100,'Fare']=100

figure=data.Age.hist(bins=50)
figure.set_title('Fare')
figure.set_xlabel('Fare')
figure.set_ylabel('No of passenger')


figure=data.Fare.hist(bins=50)
figure.set_title('Fare')
figure.set_xlabel('Fare')
figure.set_ylabel('No of passenger')


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(data[['Age','Fare']].fillna(0),data['Survived'],test_size=0.3)

### Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
y_pred1=classifier.predict_proba(X_test)

from sklearn.metrics import accuracy_score,roc_auc_score
print("Accuracy_score: {}".format(accuracy_score(y_test,y_pred)))
print("roc_auc_score: {}".format(roc_auc_score(y_test,y_pred1[:,1])))

