#!/usr/bin/env python3
# Preload packages
import pandas as pd
import numpy as np
import sklearn
import joblib
# Gridsearch runned on HPC-Cedia cluster. Hyperparameters setted to maximize accuracy and recall response. 

# Load data
bc = pd.read_csv("./Mix_BC_srbal.csv.gz")
bc_input = bc.iloc[0:466, 0:300]
bc_output = bc['Class']

# Data partition (mathematical notation)
from sklearn.model_selection import train_test_split as tts
X, Xt, y, yt = tts(bc_input,bc_output,random_state=74)

# Training and Tuning models

# RBF
# 1) gamma from 0.01 to 1 y C from 1 to 100 becomes 0.9484
param_grid = {
	'gamma': [i/100 for i in range(1,101)], 
	'C': [i for i in range(1,101)],
	'kernel': ['rbf']
}
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
gs = GridSearchCV(SVC(), param_grid, cv=5).fit(X,y) # lot of time
# training with best parameters
svmrbf = SVC(**gs.best_params_).fit(X,y)
# Export metrics
results = pd.DataFrame(gs.cv_results_)
Features = list(results.columns)
params = pd.DataFrame(results,columns=Features)
params.to_csv('./gridsearch/rbf.csv', index=False)
joblib.dump(svmrbf, "./models/bc_svmrbf.pkl")

# Linear
# 2) # 100*100 mesh from gamma: 0.01 to 1, and C from 1 to 100
# generates an accuracy of 0.91 with C: 3, gamma= 0.8
param_grid = {
	'gamma': [i/100 for i in range(1,101)], 
	'C': [i for i in range(1,101)],
	'kernel': ['linear']
}
gs = GridSearchCV(SVC(), param_grid, cv=5).fit(X,y) # takes a lot of time
# training with best parameters
svmlin = SVC(**gs.best_params_).fit(X,y)
# Export metrics
results = pd.DataFrame(gs.cv_results_)
Features = list(results.columns)
params = pd.DataFrame(results,columns=Features)
params.to_csv('./gridsearch/linear.csv', index=False)
joblib.dump(svmlin, "./models/bc_svmlin.pkl")

# Logistic regression
# 3) generates an accuracy of 0.91 with C: 3, gamma= 0.8
param_grid = {
        'C': np.logspace(-3,4,50),
        'penalty': ['l2', 'none'],
        'solver': ['newton-cg', 'lbfgs', 'sag'],
        'random_state': [74],
        'max_iter': [50000]
}
from sklearn.linear_model import LogisticRegression as LR
gs = GridSearchCV(LR(), param_grid, cv=5).fit(X,y)
# training with best parameters
lr = LR(**gs.best_params_).fit(X,y)
# Export metrics
results = pd.DataFrame(gs.cv_results_)
Features = list(results.columns)
params = pd.DataFrame(results,columns=Features)
params.to_csv('./gridsearch/logistic.csv', index=False)
joblib.dump(lr, "./models/bc_lr.pkl")
# A second option could be added with the second best lr params

# Multilayer perceptron
# 4) 
param_grid = {
        'hidden_layer_sizes': [ (20,), (100,), (50,50,50), (50,100,50)],
        'activation': ['logistic', 'tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.05],
        'learning_rate_init': np.logspace(-3,0,4),
        'random_state': [74],
        'max_iter': [50000],
        'shuffle': [False]
}
from sklearn.neural_network import MLPClassifier as MLPC
gs = GridSearchCV(MLPC(), param_grid, cv=3).fit(X,y)
# training with best parameters
mlp = MLPC(**gs.best_params_).fit(X,y)
# Export metrics
results = pd.DataFrame(gs.cv_results_)
Features = list(results.columns)
params = pd.DataFrame(results,columns=Features)
params.to_csv('./gridsearch/mlp.csv', index=False)
joblib.dump(mlp, "./models/bc_mlp.pkl")
'''
Problema dual: maximizas el valor de la ganancia
'''
# Validation
models = [svmlin, svmrbf, lr, mlp]
# K-fold Validation
from sklearn.model_selection import cross_validate as cv
from sklearn.metrics import make_scorer, accuracy_score, recall_score, roc_auc_score
scoring = ['accuracy','recall','roc_auc']
kfv = [cv(p, X,y,cv=5,scoring= scoring, n_jobs=-1) for p in models]
# K-stratified Validation
from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=74)
ksfv = [cv(p, X,y,cv=kfold,scoring= scoring, n_jobs=-1) for p in models]

metrics = list(itertools.chain.from_iterable(zip(kfv, ksfv)))

# Exporting metrics to csv
metrics = pd.concat(map(pd.DataFrame, (metrics[i] for i in range(0,len(metrics)))))
metrics['folds'] = 8*['fold'+str(i+1) for i in range(5)]
models = ['svmlin', 'svmrbf', 'lr', 'mlp']
metrics['model'] = np.append(np.repeat(models, 5),np.repeat(models, 5))
metrics['method'] = np.repeat(['kfold','stratified'],20)
metrics.to_csv('./validation_metrics.csv')

# Make predictions
yp = [svmlin.predict(Xt), svmrbf.predict(Xt), lr.predict(Xt), mlp.predict(Xt)]
# Create a classification report
metrics = pd.DataFrame()
metrics['method'] = ['svmlin', 'svmrbf', 'lr', 'mlp']
metrics['acc'] = [accuracy_score(p, yt) for p in yp]
metrics['auc'] = [roc_auc_score(p, yt) for p in yp]
metrics['recall'] = [recall_score(p, yt) for p in yp]
metrics.to_csv('./model_metrics.csv')
