#!/usr/bin/env python3
import numpy as np
import pandas as pd
import sklearn
import joblib
import imblearn

# load data, scale and resample
scaler = joblib.load("../../1_warehousing/minmax.pkl")
sel = pd.read_csv("../../1_warehousing/Mix_BC_selected.csv.gz")
X = scaler.fit_transform(sel)
y = pd.read_csv("../../1_warehousing/Mix_BC.csv.gz")['V2']

# SMOTE
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=74)
Xs, ys = smote.fit_resample(X, y)

# partition data (train:75, validation:15, test:10)
from sklearn.model_selection import train_test_split as tts
X, Xt, y, yt = tts(
	bc_input,
	bc_output,
	random_state=74,
	test_size=0.25) # 1-trainratio

Xv, Xt, yv, yt = tts(
	Xt,
	yt,
	random_state=74,
	test_size=0.4)

# Export smoted, balanced data
# X = pd.DataFrame(X, columns = sel.columns)

# train best model (stack_1)
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
# SVM
svmrbf = SVC(
		kernel='rbf',
		probability=True,
		random_state=74)

param_grid = {
	'gamma': [i/100 for i in range(1,101)], 
	'C': [i for i in range(1,101)],
	'kernel': ['rbf']
}

gs = GridSearchCV(
	SVC(), 
	param_grid,
	n_jobs=-1, 
	cv=10).fit(Xs,ys) # lot of time

# training with best parameters
svmrbf = SVC(**gs.best_params_).fit(X,y)

# MLP
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier()

param_grid = {
        'hidden_layer_sizes': [x for x in itertools.product((100,80,60,40,20,15), repeat=2)],
        'activation': ['logistic', 'relu'],
        'solver': ['adam'],
        'alpha': [0.0001,0.01],
        'learning_rate_init': np.logspace(-3,-1,11),
        'random_state': [74],
        'max_iter': [50000],
        'shuffle': [False]
}


gs = GridSearchCV(
	MLPClassifier(),
	param_grid,
	n_jobs=-1,
	cv=10).fit(X,y)

# training with best parameters
mlp = MLPClassifier(**gs.best_params_).fit(X,y)

# adaboost with rbf

adarbf = AdaBoostClassifier(
	base_estimator=SVC(
		kernel='rbf',
		probability=True,
		random_state=74))

# set parameters
param_grid = {
	'n_estimators': (1,10,25,50,100),                  
	'learning_rate': (0.0001, 0.001, 0.01, 0.1, 1.0),
	'base_estimator__C': np.logspace(-3,3,10), # aumentar  y usar logscale 1 a 1000
	'base_estimator__gamma': np.logspace(-4,2,10)} # aumentar  y usar logscale 0.0001 hasta 10

# gridsearch
gs=GridSearchCV(
	adarbf,
	param_grid,
	n_jobs=-1,
	cv=10).fit(X,y) # lot of time	

adarbf = AdaBoostClassifier(**gs.best_params_).fit(X,y)

from sklearn.ensemble import StackingClassifier

estimators = [("svm", svmrbf),("mlp",mlp)]
stack_1 = StackingClassifier(estimators = estimators, final_estimator = lr).fit(X, y)

joblib.dump(stack_1, "selected.pkl")




