# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 17:21:53 2020

@author: Vandan
"""

################################################################################################

# Handling Missing Values

################################################################################################

df=pd.read_csv('titanic.csv')

# =============================================================================
#MCAR

# Missing Completely at Random, MCAR: A variable is missing completely at random (MCAR) 
#if the probability of being missing is the same for all the observations. When data is MCAR,
# there is absolutely no relationship between the data missing and any other values, observed 
# or missing, within the dataset. In other words, those missing data points are a random subset 
# of the data. There is nothing systematic going on that makes some data more likely to be
# missing than other.
# =============================================================================


df.isnull().sum()
# Here embarked is considered as mcar


# =============================================================================
#MNAR

# #Missing Data Not At Random(MNAR): Systematic missing Values There is absolutely some
# # relationship between the data missing and any other values, observed or missing, within 
# # the dataset.
# 
# =============================================================================

import numpy as np
df['cabin_null']=np.where(df['Cabin'].isnull(),1,0)

##find the percentage of null values
df['cabin_null'].mean()
df.groupby(['Survived'])['cabin_null'].mean()


# =============================================================================
#MAR


#Men---hide their salary
#Women---hide their age
# 
# =============================================================================

# Check whether null values and others values



# Techniques to handle missing data


#1. Mean/ MEdian /Mode imputation

# When should we apply? 
# Mean/median imputation has the assumption that the data are 
# missing completely at random(MCAR). We solve this by replacing the NAN with the most
# frequent occurance of the variables

def impute_nan(df,variable,median):
    df[variable+"_median"]=df[variable].fillna(median)

impute_nan(df,'Age',median)

# Check the standard deviation before and after imputation
print(df['Age'].std())
print(df['Age_median'].std())


#visualize distribution
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
df['Age'].plot(kind='kde', ax=ax)
df.Age_median.plot(kind='kde', ax=ax, color='red')
lines, labels = ax.get_legend_handles_labels()
ax.legend(lines, labels, loc='best')

#Advantages
#Easy to implement(Robust to outliers)
#Faster way to obtain the complete dataset 

#### Disadvantages
#Change or Distortion in the original variance
#Impacts Correlation




# 2. Random Sample imputation
#Aim: Random sample imputation consists of taking random observation from the
# dataset and we use this observation to replace the nan values
# When should it be used?
# It assumes that the data are missing completely at random(MCAR)

Data = pd.read_csv('titanic.csv')
Data['Age'].dropna().sample(Data['Age'].isnull().sum())



def impute_nan(df,variable,median):
    df[variable+"_median"]=df[variable].fillna(median)
    df[variable+"_random"]=df[variable]
    ##It will have the random sample to fill the na
    random_sample=df[variable].dropna().sample(df[variable].isnull().sum(),random_state=0)
    ##pandas need to have same index in order to merge the dataset
    random_sample.index=df[df[variable].isnull()].index
    df.loc[df[variable].isnull(),variable+'_random']=random_sample


impute_nan(df,"Age",median)


fig = plt.figure()
ax = fig.add_subplot(111)
df['Age'].plot(kind='kde', ax=ax)
df.Age_median.plot(kind='kde', ax=ax, color='red')
df.Age_random.plot(kind='kde', ax=ax, color='green')
lines, labels = ax.get_legend_handles_labels()
ax.legend(lines, labels, loc='best')


#Advantages
#Easy To implement
#There is less distortion in variance

#Disadvantage
#Every situation randomness wont work





# 3. Capturing NAN values with a new feature
# It works well if the data are not missing completely at random
# It creates additional feature to give importance to values those are missing. 
# Here, Model will be able to learn something about missing values.

df=pd.read_csv('titanic.csv', usecols=['Age','Fare','Survived'])
df.head()
import numpy as np
df['Age_NAN']=np.where(df['Age'].isnull(),1,0)

df['Age'].fillna(df.Age.median(),inplace=True)


#Advantages
#Easy to implement
#Captures the importance of missing values
#Disadvantages
#Creating Additional Features(Curse of Dimensionality)



# 4. End of Distribution imputation
# Missing values are replaced with value after 3 standard deviation
df=pd.read_csv('titanic.csv', usecols=['Age','Fare','Survived'])
df.head()

extreme=df.Age.mean()+3*df.Age.std()

import seaborn as sns
sns.boxplot('Age',data=df)

def impute_nan(df,variable,median,extreme):
    df[variable+"_end_distribution"]=df[variable].fillna(extreme)
    df[variable].fillna(median,inplace=True)

#box plots and histograms
import seaborn as sns

sns.boxplot('Age',data=df)

df['Age'].hist(bins=50)

df['Age_end_distribution'].hist(bins=50)

sns.boxplot('Age_end_distribution',data=df)

# Advantges
# It gives proper distribution of variable    
# captures importance of missingness

#Disadvantages
# distorts the original distribution of the variable
# if missingness is not important, it may mask the predictive power of original variable by distorting its distribution
# if the number of NA is big, it will mask true outliers in the distribution 
# if the number os NA is small, the replaced NA may be considered an outlier and prepocessed in a subsequent step of feature engineering





# 5. Arbitary feature imputation
# Arbitary valye used here should not be frequently occuring
# use values at completely at the end

def impute_nan(df,variable):
    df[variable+"_hundred"]=df[variable].fillna(100)

#Advantage 
# easy to use

# Disadvantage
# Hard to decide which value to use




# 6. Replace it with most frequent value (Categorical variable) 
def impute_nan(df,variable):
    most_frequent = df[variable].mode()[0]
    df[variable]=df[variable].fillna(most_frequent)
    

for feature in ['abc','cbd']:
    impute_nan(df,feature)

# Advantages
# Easy to use

# Disadvantage
# If more missing values, it may use them in overrepresented way    
# It distorts the relation of the most frequent label  
    
 
    
#  7. create new feature  respect to missing values (Categorical variable)
    
df=pd.read_csv('titanic.csv', usecols=['Embarked','Survived'])
df.head()
import numpy as np
df['Embarked_NAN']=np.where(df['Embarked'].isnull(),1,0)

df['Embarked'].fillna(df.Embarked.mode[0],inplace=True)




#  8. insert new category where there is missing values

df['Embarked'].fillna('missing',inplace=True)

