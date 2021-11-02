# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 15:33:37 2020

@author: Vandan
"""

# Hyperparameter tuning


#All Techniques Of Hyper Parameter Optimization
#GridSearchCV
#RandomizedSearchCV
#Bayesian Optimization -Automate Hyperparameter Tuning (Hyperopt)
#Sequential Model Based Optimization(Tuning a scikit-learn estimator with skopt)
#Optuna- Automate Hyperparameter Tuning
#Genetic Algorithms (TPOT Classifier)


# Refrences

#https://github.com/fmfn/BayesianOptimization
#https://github.com/hyperopt/hyperopt
#https://www.jeremyjordan.me/hyperparameter-tuning/
#https://optuna.org/
#https://towardsdatascience.com/hyperparameters-optimization-526348bb8e2d(By Pier Paolo Ippolito )
#https://scikit-optimize.github.io/stable/auto_examples/hyperparameter-optimization.html





# It is useful when you have large amount of data



import warnings
warnings.filterwarnings('ignore')


import pandas as pd
df=pd.read_csv('diabetes.csv')
df.head()



import numpy as np
df['Glucose']=np.where(df['Glucose']==0,df['Glucose'].median(),df['Glucose'])
df['Insulin']=np.where(df['Insulin']==0,df['Insulin'].median(),df['Insulin'])
df['SkinThickness']=np.where(df['SkinThickness']==0,df['SkinThickness'].median(),df['SkinThickness'])
df.head()


#### Independent And Dependent features
X=df.drop('Outcome',axis=1)
y=df['Outcome']


#### Train Test Split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=42)

X_train.head()

from sklearn.ensemble import RandomForestClassifier
rf_classifier=RandomForestClassifier(n_estimators=10).fit(X_train,y_train)
prediction=rf_classifier.predict(X_test)


from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
print(confusion_matrix(y_test,prediction))
print(accuracy_score(y_test,prediction))
print(classification_report(y_test,prediction))





# Randomized search cv
# It randomnly takes values of parameters and applies to it.

from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt','log2']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 1000,10)]
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10,14]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4,6,8]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
              'criterion':['entropy','gini']}
print(random_grid)


# create classifier
rf=RandomForestClassifier()
rf_randomcv=RandomizedSearchCV(estimator=rf,param_distributions=random_grid,n_iter=100,cv=3,verbose=2,
                               random_state=100,n_jobs=-1)

### fit the randomized model
rf_randomcv.fit(X_train,y_train)

# Provides best parameters
rf_randomcv.best_params_


best_random_grid=rf_randomcv.best_estimator_

from sklearn.metrics import accuracy_score
y_pred=best_random_grid.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print("Accuracy Score {}".format(accuracy_score(y_test,y_pred)))
print("Classification report: {}".format(classification_report(y_test,y_pred)))







# Grid search cv
# It check one by one values of parameters
# It is recommended to first use randomized search cv and then grid search cv
# First take the best values of parameters from randomized search cv and then choose values around it
# iteration = multiplication of numbers of values of parameters

from sklearn.model_selection import GridSearchCV

param_grid = {
    'criterion': [rf_randomcv.best_params_['criterion']],
    'max_depth': [rf_randomcv.best_params_['max_depth']],
    'max_features': [rf_randomcv.best_params_['max_features']],
    'min_samples_leaf': [rf_randomcv.best_params_['min_samples_leaf'], 
                         rf_randomcv.best_params_['min_samples_leaf']+2, 
                         rf_randomcv.best_params_['min_samples_leaf'] + 4],
    'min_samples_split': [rf_randomcv.best_params_['min_samples_split'] - 2,
                          rf_randomcv.best_params_['min_samples_split'] - 1,
                          rf_randomcv.best_params_['min_samples_split'], 
                          rf_randomcv.best_params_['min_samples_split'] +1,
                          rf_randomcv.best_params_['min_samples_split'] + 2],
    'n_estimators': [rf_randomcv.best_params_['n_estimators'] - 200, rf_randomcv.best_params_['n_estimators'] - 100, 
                     rf_randomcv.best_params_['n_estimators'], 
                     rf_randomcv.best_params_['n_estimators'] + 100, rf_randomcv.best_params_['n_estimators'] + 200]
}

print(param_grid)
    


#### Fit the grid_search to the data
rf=RandomForestClassifier()
grid_search=GridSearchCV(estimator=rf,param_grid=param_grid,cv=10,n_jobs=-1,verbose=2)
grid_search.fit(X_train,y_train)

grid_search.best_estimator_

best_grid=grid_search.best_estimator_


y_pred=best_grid.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print("Accuracy Score {}".format(accuracy_score(y_test,y_pred)))
print("Classification report: {}".format(classification_report(y_test,y_pred)))







# Automated hyperparameter tuning
# Automated Hyperparameter Tuning can be done by using techniques such as
#
# 1) Bayesian Optimization
# 2) Gradient Descent
# 3) Evolutionary Algorithms





# Bayesian Optimization
# Bayesian optimization uses probability to find the minimum of a function.
# The final aim is to find the input value to a function which can gives us the lowest
# possible output value.It usually performs better than random,grid and manual search
# providing better performance in the testing phase and reduced optimization time.
# In Hyperopt, Bayesian Optimization can be implemented giving 3 three main parameters 
# to the function fmin.
#
#Objective Function = defines the loss function to minimize.

#Domain Space = defines the range of input values to test 
#(in Bayesian Optimization this space creates a probability distribution for 
#each of the used Hyperparameters).

# Optimization Algorithm = defines the search algorithm to use to select the best
# input values to use in each new iteration.

from hyperopt import hp,fmin,tpe,STATUS_OK,Trials
from sklearn.model_selection import cross_val_score


# Creating Space
space = {'criterion': hp.choice('criterion', ['entropy', 'gini']),
        'max_depth': hp.quniform('max_depth', 10, 1200, 10),
        'max_features': hp.choice('max_features', ['auto', 'sqrt','log2', None]),
        'min_samples_leaf': hp.uniform('min_samples_leaf', 0, 0.5),
        'min_samples_split' : hp.uniform ('min_samples_split', 0, 1),
        'n_estimators' : hp.choice('n_estimators', [10, 50, 300, 750, 1200,1300,1500])
    }


space


# Objective function
def objective(space):
    model = RandomForestClassifier(criterion = space['criterion'], max_depth = space['max_depth'],
                                 max_features = space['max_features'],
                                 min_samples_leaf = space['min_samples_leaf'],
                                 min_samples_split = space['min_samples_split'],
                                 n_estimators = space['n_estimators'], 
                                 )
    
    accuracy = cross_val_score(model, X_train, y_train, cv = 5).mean()

    # We aim to maximize accuracy, therefore we return it as a negative value
    return {'loss': -accuracy, 'status': STATUS_OK }



trials = Trials()
best = fmin(fn= objective,
            space= space,
            algo= tpe.suggest,
            max_evals = 80,
            trials= trials)
best

# take key value pairs
crit = {0: 'entropy', 1: 'gini'}
feat = {0: 'auto', 1: 'sqrt', 2: 'log2', 3: None}
est = {0: 10, 1: 50, 2: 300, 3: 750, 4: 1200,5:1300,6:1500}


print(crit[best['criterion']])
print(feat[best['max_features']])
print(est[best['n_estimators']])

best['min_samples_leaf']

trainedforest = RandomForestClassifier(criterion = crit[best['criterion']], max_depth = best['max_depth'], 
                                       max_features = feat[best['max_features']], 
                                       min_samples_leaf = best['min_samples_leaf'], 
                                       min_samples_split = best['min_samples_split'], 
                                       n_estimators = est[best['n_estimators']]).fit(X_train,y_train)
predictionforest = trainedforest.predict(X_test)
print(confusion_matrix(y_test,predictionforest))
print(accuracy_score(y_test,predictionforest))
print(classification_report(y_test,predictionforest))
acc5 = accuracy_score(y_test,predictionforest)







# Genetic Algorithm

#Genetic Algorithms tries to apply natural selection mechanisms to Machine Learning contexts.

# Let's immagine we create a population of N Machine Learning models with some 
# predifined Hyperparameters. We can then calculate the accuracy of each model
# and decide to keep just half of the models (the ones that performs best). 
# We can now generate some offsprings having similar Hyperparameters to the ones 
# of the best models so that go get again a population of N models. 
# At this point we can again caltulate the accuracy of each model and repeate the
# cycle for a defined number of generations. 
#In this way, just the best models will survive at the end of the process.



import numpy as np
from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt','log2']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 1000,10)]
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10,14]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4,6,8]
# Create the random grid
param = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
              'criterion':['entropy','gini']}
print(param)


#Installl tensorflow and tpot

from tpot import TPOTClassifier


tpot_classifier = TPOTClassifier(generations= 5, population_size= 24, offspring_size= 12,
                                 verbosity= 2, early_stop= 12,
                                 config_dict={'sklearn.ensemble.RandomForestClassifier': param}, 
                                 cv = 4, scoring = 'accuracy')
tpot_classifier.fit(X_train,y_train)


accuracy = tpot_classifier.score(X_test, y_test)
print(accuracy)





# Optuna

# Optimize hyperparameters of the model using Optuna
# The hyperparameters of the above algorithm are n_estimators 
# and max_depth for which we can try different values to see 
# if the model accuracy can be improved. 
# The objective function is modified to accept a trial object.
# This trial has several methods for sampling hyperparameters. 
# We create a study to run the hyperparameter optimization and 
# finally read the best hyperparameters.


import optuna
import sklearn.svm
def objective(trial):
    
    classifier = trial.suggest_categorical('classifier', ['RandomForest', 'SVC'])
    
    # Specify more classifiers
    if classifier == 'RandomForest':
        # mention more parameters
        n_estimators = trial.suggest_int('n_estimators', 200, 2000,10)
        max_depth = int(trial.suggest_float('max_depth', 10, 100, log=True))

        clf = sklearn.ensemble.RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth)
    else:
        c = trial.suggest_float('svc_c', 1e-10, 1e10, log=True)
        
        clf = sklearn.svm.SVC(C=c, gamma='auto')

    return sklearn.model_selection.cross_val_score(
        clf,X_train,y_train, n_jobs=-1, cv=3).mean()


study = optuna.create_study(direction='maximize') # maximize= accuracy and minimze = loss
study.optimize(objective, n_trials=100)

trial = study.best_trial

print('Accuracy: {}'.format(trial.value))
print("Best hyperparameters: {}".format(trial.params))


trial


study.best_params


rf=RandomForestClassifier(n_estimators=330,max_depth=30)
rf.fit(X_train,y_train)



y_pred=rf.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))


# Plotting the optimization history of the study.
optuna.visualization.plot_optimization_history(study)

# Plotting the accuracies for each hyperparameter for each trial.
optuna.visualization.plot_slice(study)

# Plotting the accuracy surface for the hyperparameters involved in the random forest model.
optuna.visualization.plot_contour(study, params=['n_estimators', 'max_depth'])
