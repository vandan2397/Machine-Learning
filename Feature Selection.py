# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 15:27:11 2020

@author: Vandan
"""

# If any duplicate rows or column are found, then delete it

# keep first duplicate row
# result_df = source_df.drop_duplicates()

import pandas as pd
df=pd.read_csv('mobile_dataset.csv')
df.head()


# Univariate selection

X=df.iloc[:,:-1]
y=df['price_range']

# Feature selection using SelectKBest
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

ordered_rank_features=SelectKBest(score_func=chi2,k=20)
ordered_feature=ordered_rank_features.fit(X,y)

dfscores=pd.DataFrame(ordered_feature.scores_,columns=["Score"])
dfcolumns=pd.DataFrame(X.columns)

features_rank=pd.concat([dfcolumns,dfscores],axis=1)
features_rank.columns=['Features','Score']
features_rank




# Feature selection using when independent features and dependent features are numerical 

from sklearn.feature_selection import f_regression

# define feature selection
fs = SelectKBest(score_func=f_regression, k=10)
# apply feature selection
ordered_feature=ordered_rank_features.fit(X,y)

dfscores=pd.DataFrame(ordered_feature.scores_,columns=["Score"])
dfcolumns=pd.DataFrame(X.columns)

features_rank=pd.concat([dfcolumns,dfscores],axis=1)
features_rank.columns=['Features','Score']
features_rank




# # Feature selection using when independent features are numerical 
# and dependent feature is categorical
from sklearn.feature_selection import f_classif

# define feature selection
fs = SelectKBest(score_func=f_classif, k=10)
# apply feature selection
ordered_feature=ordered_rank_features.fit(X,y)

dfscores=pd.DataFrame(ordered_feature.scores_,columns=["Score"])
dfcolumns=pd.DataFrame(X.columns)

features_rank=pd.concat([dfcolumns,dfscores],axis=1)
features_rank.columns=['Features','Score']
features_rank






# Feature importance

# Using Extratrees
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model=ExtraTreesClassifier()
model.fit(X,y)

print(model.feature_importances_)

ranked_features=pd.Series(model.feature_importances_,index=X.columns)
ranked_features.nlargest(10).plot(kind='barh')
plt.show()


# Using XGBoost
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
model=XGBClassifier()
model.fit(X,y)

print(model.feature_importances_)

ranked_features=pd.Series(model.feature_importances_,index=X.columns)
ranked_features.nlargest(10).plot(kind='barh')
plt.show()




# Correlation
df.corr()

import seaborn as sns
corr=df.iloc[:,:-1].corr()
top_features=corr.index
plt.figure(figsize=(20,20))
sns.heatmap(df[top_features].corr(),annot=True)


# Remove correlated features to avoid multi collinearity
threshold=0.8


# find and remove correlated features
def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr


correlation(df.iloc[:,:-1],threshold)





# Information gain

# Mutual information is calculated between two variables and measures the 
# reduction in uncertainty for one variable given a known value of the other variable.

from sklearn.feature_selection import mutual_info_classif
mutual_info=mutual_info_classif(X,y)
mutual_data=pd.Series(mutual_info,index=X.columns)
mutual_data.sort_values(ascending=False)





# Feature Importance using linear regression

from sklearn.linear_model import LinearRegression
from matplotlib import pyplot
# define the model
model = LinearRegression()
# fit the model
model.fit(X, y)
# get importance
importance = model.coef_
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()





# Permutation Feature Importance for Classification

#Permutation Feature Importance
#Permutation feature importance is a technique for calculating relative 
#importance scores that is independent of the model used.
#
#First, a model is fit on the dataset, such as a model that does
# not support native feature importance scores. Then the model is 
# used to make predictions on a dataset, although the values of a
# feature (column) in the dataset are scrambled. This is repeated
# for each feature in the dataset. Then this whole process is repeated
# 3, 5, 10 or more times. The result is a mean importance score for each 
# input feature (and distribution of scores given the repeats)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import permutation_importance
model = KNeighborsClassifier()
# fit the model
model.fit(X, y)
# perform permutation importance
results = permutation_importance(model, X, y, scoring='accuracy')
# get importance
importance = results.importances_mean
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()



   



# Feature selection using Recursive feature selection
#feature selection using recursive feature selection technique
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
rfc = RandomForestClassifier(random_state=0,n_estimators=1000,criterion="gini")

selector = RFECV(estimator=rfc, step=1, cv=StratifiedKFold(10),
              scoring='accuracy')
selector.fit(X, y)

print("Optimal number of features : %d" % selector.n_features_)

# Plot number of features VS. cross-validation scores
plt.figure(figsize=(15,8))
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score")
plt.plot(range(1, len(selector.grid_scores_) + 1), selector.grid_scores_)
plt.show()

for i in range(X.shape[1]):
	print('Column: %d, Selected %s, Rank: %.3f' % (i, selector.support_[i], selector.ranking_[i]))

 
    
    
    
# Variance threshold
# It drops columns having variance 0


### It will zero variance features
from sklearn.feature_selection import VarianceThreshold
var_thres=VarianceThreshold(threshold=0)
var_thres.fit(X_train)    
    
### Finding non constant features
sum(var_thres.get_support())


# Lets Find non-constant features 
len(X_train.columns[var_thres.get_support()])

constant_columns = [column for column in X_train.columns if column not in X_train.columns[var_thres.get_support()]]
print(len(constant_columns))

X_train.drop(constant_columns,axis=1)





