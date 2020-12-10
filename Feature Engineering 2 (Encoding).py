# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 15:15:06 2020

@author: Vandan
"""

import numpy as np
import pandas as pd



################################################################################################

# Encoding

################################################################################################

# One Hot Encoding (Nominal categories)
# Creates dummy variables
# Using pandas
Data = pd.read_csv('titanic.csv',usecols=['Sex','Embarked'])
dummy_variable = pd.get_dummies(Data, drop_first=True)




# Multiple categories
# You can use top 10 most occuring categories
df=pd.read_csv('mercedes.csv',usecols=["X0","X1","X2","X3","X4","X5","X6"]) 

for i in df.columns:
    print(len(df[i].unique()))
    
lst_10=df.X1.value_counts().sort_values(ascending=False).head(10).index
lst_10=list(lst_10)    
print(lst_10)    
 
for categories in lst_10:
    df[categories]=np.where(df['X1']==categories,1,0)   
    



# Label Encoding (Ordinal categories)
# Assign ranks (integer)
import datetime
today_date=datetime.datetime.today()
days=[today_date-datetime.timedelta(x) for x in range(0,15)]
data=pd.DataFrame(days)
data.columns=["Day"]

data['weekday']=data['Day'].dt.weekday_name
data.head()
dictionary={'Monday':1,'Tuesday':2,'Wednesday':3,'Thursday':4,'Friday':5,'Saturday':6,'Sunday':7}
data['weekday_ordinal']=data['weekday'].map(dictionary)




# Count or frquency encoding
# replaces category with frequency
data2 = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data')
columns=[1,3,5,6,7,8,9,13]
data2=data2[columns]
data2.columns=['Employment','Degree','Status','Designation','family_job','Race','Sex','Country']
country_map=data2['Country'].value_counts().to_dict()
data2['Country']=data2['Country'].map(country_map)



# Mean Encoding (Ordinal categories)
# Calculates mean for each category, sorts based on mean values, and assigns order
Data=pd.read_csv('titanic.csv', usecols=['Cabin','Survived'])
Data['Cabin'].fillna('Missing',inplace=True)
Data['Cabin']=Data['Cabin'].astype(str).str[0]
Data.groupby(['Cabin'])['Survived'].mean()
ordinal_labels=Data.groupby(['Cabin'])['Survived'].mean().sort_values().index
ordinal_labels2={k:i for i,k in enumerate(ordinal_labels,0)}
ordinal_labels2
Data['Cabin_ordinal_labels']=Data['Cabin'].map(ordinal_labels2)
Data.head()


# Target Encoding (Nominal categories)
# Calculates mean for each category and replace it with category
Data=pd.read_csv('titanic.csv', usecols=['Cabin','Survived'])
Data['Cabin'].fillna('Missing',inplace=True)
Data['Cabin']=Data['Cabin'].astype(str).str[0]
Data.groupby(['Cabin'])['Survived'].mean()
mean_ordinal=df.groupby(['Cabin'])['Survived'].mean().to_dict()
Data['mean_ordinal_encode']=Data['Cabin'].map(mean_ordinal)
Data.head()




# Probablity ratio encoding
# Calculates pobablity ratio using probablities of both categories of target value
Data=pd.read_csv('titanic.csv', usecols=['Cabin','Survived'])
Data['Cabin'].fillna('Missing',inplace=True)
Data['Cabin']=Data['Cabin'].astype(str).str[0]
prob_df=Data.groupby(['Cabin'])['Survived'].mean()
prob_df=pd.DataFrame(prob_df)
prob_df['Died']=1-prob_df['Survived']
prob_df['Probability_ratio']=prob_df['Survived']/prob_df['Died']
probability_encoded=prob_df['Probability_ratio'].to_dict()
Data['Cabin_encoded']=Data['Cabin'].map(probability_encoded)









