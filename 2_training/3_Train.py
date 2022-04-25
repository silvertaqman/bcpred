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
params.to_csv('rbf.csv', index=False)

# Linear
# 2) # 100*100 mesh from gamma: 0.01 to 1, and C from 1 to 100
# generates an accuracy of 0.91 with C: 3, gamma= 0.8
param_grid = {
	'gamma': [i/100 for i in range(1,101)], 
	'C': [i for i in range(1,101)],
	'kernel': ['linear']
}
param_grid = {'C': C,'gamma': gamma, 'kernel':kernel}
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
gs = GridSearchCV(SVC(), param_grid, cv=5).fit(X,y) # takes a lot of time
# training with best parameters
svmrbf = SVC(**gs.best_params_).fit(X,y)
# Export metrics
results = pd.DataFrame(gs.cv_results_)
Features = list(results.columns)
params = pd.DataFrame(results,columns=Features)
params.to_csv('linear.csv', index=False)

# Logistic regression
# 3) generates an accuracy of 0.91 with C: 3, gamma= 0.8
penalty = 
C = 
solver = 
param_grid = {
	'C': np.logspace(-3,3,7),
	'penalty': ['l2', 'none'], 
	'solver': ['newton-cg', 'lbfgs', 'sag'], 
	multiclass = 'ovr'
}
from sklearn.linear_model import LogisticRegression as LR
gs = GridSearchCV(LR(), param_grid, cv=5).fit(X,y)
# training with best parameters
lr = LR(**gs.best_params_).fit(X,y)
# Export metrics
results = pd.DataFrame(gs.cv_results_)
Features = list(results.columns)
params = pd.DataFrame(results,columns=Features)
params.to_csv('logistic.csv', index=False)
# A second option could be added with the second best lr params

# Multilayer perceptron
# 4) 

param_grid = {
	'hidden_layer_sizes': [ (20,), (50,50,50), (50,100,50), (100,)],
	'activation': ['identity', 'logistic', 'tanh', 'relu'],
	'solver': ['sgd', 'adam'],
	'alpha': [0.0001, 0.001, 0.05],
	'learning_rate_init': np.logspace(-6,0,7),
	'random_state' = 74, 
	'max_iter'= 50000,
	'shuffle'= False
}
from sklearn.neural_network import MLPClassifier as MLPC
gs = GridSearchCV(MLPC(), param_grid, cv=5).fit(X,y)
# training with best parameters
mlp = MLPC(**gs.best_params_).fit(X,y)
# Export metrics
results = pd.DataFrame(gs.cv_results_)
Features = list(results.columns)
params = pd.DataFrame(results,columns=Features)
params.to_csv('mlp.csv', index=False)
'''
Problema dual: maximizas el valor de la ganancia
'''
# Export models
joblib.dump(svmlin, "./models/bc_svmlin.pkl")
joblib.dump(svmrbf, "./models/bc_svmrbf.pkl")
joblib.dump(lr, "./models/bc_lr.pkl")
joblib.dump(mlp, "./models/bc_mlp.pkl")

# Validation
models = [svmlin, svmrbf, lr, mlp]
# K-fold Validation
from sklearn.model_selection import cross_val_score as cvs
kfv = [cvs(p, X,y,cv=5) for p in models]
# K-stratified Validation
from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=74)
ksfv = [cvs(p, X,y,cv=kfold) for p in models]
cols = ['fold'+str(i) for i in range(1,6)]
# Export results
metrics = pd.DataFrame(kfv+ksfv, columns = cols)
metrics['validation'] = 4*['kfold']+4*['strat']
metrics['mean_acc'] = [np.mean(i) for i in kfv+ksfv]
metrics['mean_sd'] = [np.std(i) for i in kfv+ksfv]
metrics['method'] = 2*['svmlin', 'svmrbf', 'lr', 'mlp']
metrics.to_csv('./validation_metrics.csv')

# Metrics (Every model)
# Make predictions
yp = [svmlin.predict(Xt), svmrbf.predict(Xt), lr.predict(Xt), mlp.predict(Xt)]
# Create a classification report
metrics = pd.DataFrame()
metrics['method'] = ['svmlin', 'svmrbf', 'lr', 'mlp']
metrics['acc'] = [accuracy_score(p, yt) for p in yp]
metrics['auc'] = [roc_auc_score(p, yt) for p in yp]
metrics['recall'] = [recall_score(p, yt) for p in yp]
metrics.to_csv('./model_metrics.csv')
