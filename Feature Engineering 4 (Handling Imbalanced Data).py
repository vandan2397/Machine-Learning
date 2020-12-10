# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 16:26:26 2020

@author: Vandan
"""

################################################################################################

# Handling Imbalanced Data

################################################################################################
import pandas as pd
import numpy as np
df = pd.read_csv('creditcard.csv')

df.head()


df['Class'].value_counts()

x = df.iloc[:,0:30]
y = df.iloc[:,30]


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.7)




# 1. KFOLD and hyperparameter tuning

log_class = LogisticRegression()
grid = {'C':10.0 **np.arange(-2,3),'penalty':['l2']}
cv = KFold(n_splits=5, random_state=None,shuffle=False)

clf = GridSearchCV(log_class,grid,cv=cv,n_jobs=-1,scoring='f1_macro')
clf.fit(x_train,y_train)

y_pred = clf.predict(x_test)
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))




# 2. By assigning weights (giving more importance to less frequent class)
class_weight = dict({0:1,1:100})

from sklearn.ensemble.forest import RandomForestClassifier
classifier = RandomForestClassifier(class_weight=class_weight)
classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))




# 3. Undersampling
from collections import Counter
from imblearn.under_sampling import NearMiss
ns = NearMiss(0.8)
x_train_new, y_train_new = ns.fit_sample(x_train,y_train)    

print('Number of classes before: {}'.format(Counter(y_train)))
print('Number of classes after: {}'.format(Counter(y_train_new)))

classifier = RandomForestClassifier()
classifier.fit(x_train_new,y_train_new)

y_pred = classifier.predict(x_test)
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))



# 4. Oversampling
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
os = RandomOverSampler(0.5)
x_train_os, y_train_os = os.fit_sample(x_train,y_train)    

print('Number of classes before: {}'.format(Counter(y_train)))
print('Number of classes after: {}'.format(Counter(y_train_os)))

classifier = RandomForestClassifier()
classifier.fit(x_train_os,y_train_os)

y_pred = classifier.predict(x_test)
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))


# 5. SMOTETomek
from imblearn.combine import SMOTETomek
ros = RandomOverSampler(0.75)
x_train_ros, y_train_ros = ros.fit_sample(x_train,y_train)    
print('Number of classes before: {}'.format(Counter(y_train)))
print('Number of classes after: {}'.format(Counter(y_train_os)))

classifier = RandomForestClassifier()
classifier.fit(x_train_ros,y_train_ros)

y_pred = classifier.predict(x_test)
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))


# 6. Ensemble Techniques
from imblearn.ensemble import EasyEnsembleClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
# define model
model = EasyEnsembleClassifier(n_estimators=10)
# define evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(model, x_train, y_train, scoring='roc_auc', cv=cv, n_jobs=-1)
# summarize performance
print('Mean ROC AUC {}'.format(scores.mean()))

model.fit(x_train,y_train)
y_pred = model.predict(x_test)
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))

