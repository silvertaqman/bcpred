#!/usr/bin/env python3
###################################################################
# Preload packages
###################################################################
import sys
import pandas as pd
import numpy as np
import scipy
<<<<<<< HEAD
=======
import IPython
>>>>>>> 04b0460809710406095b4e1a427dc0fcdc9198cd
import sklearn
import joblib
import itertools
<<<<<<< HEAD
# Gridsearch runned on HPC-cedia cluster. Hyperparameters setted to maximize accuracy and recall responses. 
# Load data
bc = pd.read_csv("./Mix_BC_srbal.csv.gz")
bc_input = bc.iloc[0:466, 1:276]
=======
# Gridsearch runned on HPC-Cedia cluster. Hyperparameters setted to maximize accuracy and recall responses. 

# Load data
bc = pd.read_csv("../1_warehousing/Mix_BreastCancer_srbal.csv.gz")
bc_input = bc.iloc[0:466, 0:356]
>>>>>>> 04b0460809710406095b4e1a427dc0fcdc9198cd
bc_output = bc['Class']

# Metrics (Every model)
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score, f1_score

# Data partition (mathematical notation) at 75:15:10
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
	test_size=0.4) #70:20:10 # testratio/(testratio+validationratio)

<<<<<<< HEAD
###################################################################
# Training and Tuning strong models
###################################################################

# RBF
=======
F = list(X.columns)
# Selected features
with open("Selected_Features.txt","w") as f:
	[f.write("%s\n" % i) for i in F]


# Training and Tuning models

# RBF
# 1) gamma from 0.01 to 1 y C from 1 to 100 becomes 0.9484
>>>>>>> 04b0460809710406095b4e1a427dc0fcdc9198cd
param_grid = {
	'gamma': [i/100 for i in range(1,101)], 
	'C': [i for i in range(1,101)],
	'kernel': ['rbf']
}
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
gs_svmrbf = GridSearchCV(
	SVC(),
	param_grid,
	n_jobs=-1,
	cv=10).fit(X,y) # lot of time
# training with best parameters
<<<<<<< HEAD
joblib.dump(gs_svmrbf, "./gridsearch/gs_svmrbf.pkl")
pd.DataFrame(gs_svmrbf.cv_results_).to_csv("./gridsearch/svm.csv")
svmrbf = SVC(C=3,gamma=0.06,kernel="rbf",probability=True).fit(X,y)
joblib.dump(svmrbf, "./models/bc_svmrbf.pkl")

# Logistic regression
param_grid = {
        'C': np.logspace(-4, 4, 100),
        'penalty': ['none', 'l2'],
        'solver': ['newton-cg', 'lbfgs', 'sag'],
        'random_state': [74],
        'max_iter': [5000]
=======
joblib.dump(gs_svmrbf, "../gridsearch/gs_svmrbf.pkl")
svmrbf = SVC(**gs_svmrbf.best_params_,probability=True).fit(X,y)
svmrbf = CalibratedClassifierCV(svmrbf).fit(X,y
joblib.dump(svmrbf, "../models/bc_svmrbf.pkl")

# Logistic regression
# 2) generates an accuracy of 0.91 with C: 3, gamma= 0.8
param_grid = {
        'C': np.logspace(-3,4,50),
        'penalty': ['l2'],
        'solver': ['newton-cg', 'lbfgs', 'sag'],
        'random_state': [74],
        'max_iter': [10000]
>>>>>>> 04b0460809710406095b4e1a427dc0fcdc9198cd
}
from sklearn.linear_model import LogisticRegression
gs_lr = GridSearchCV(
	LogisticRegression(),
	param_grid, 
<<<<<<< HEAD
	n_jobs=-1, 
	cv=10).fit(X,y)
joblib.dump(gs_lr, "./gridsearch/gs_lr.pkl")
pd.DataFrame(gs_lr.cv_results_).to_csv("./gridsearch/lr.csv")
lr = LogisticRegression(C=0.268269579527973,penalty="l2",max_iter=5000,random_state=74,solver="lbfgs").fit(X,y)
joblib.dump(lr, "./models/bc_lr.pkl")
# Multilayer perceptron
# 3) 
hlayer = tuple(np.arange(300,701,100))
param_grid = {
        'hidden_layer_sizes': [x for x in itertools.product(hlayer, repeat=2)],
        'activation': ['relu'],
        'solver': ['adam','sgd'],
        'alpha': [0.001,0.01,0.1],
        'learning_rate_init': np.logspace(-3,-1,3),
=======
	n_jobs=-1,
	cv=10).fit(X,y)
joblib.dump(gs_lr, "./gridsearch/gs_lr.pkl")
# training with best parameters
lr = LogisticRegression(**gs_lr.best_params_).fit(X,y)
joblib.dump(lr, "./models/bc_lr.pkl")
# A second option could be added with the second best lr params
# Multilayer perceptron
# 3) 

'''
hidden_layer = [x for x in itertools.product((128*4,128*3,128*2,128*1,128/2,128/4,128/8), repeat=2)]  # repeat indica el numero de capas.
'''
param_grid = {
        'hidden_layer_sizes': [x for x in itertools.product((100,80,60,40,20,15), repeat=2)],
        'activation': ['logistic', 'relu'],
        'solver': ['adam'],
        'alpha': [0.0001,0.01],
        'learning_rate_init': np.logspace(-3,-1,11),
>>>>>>> 04b0460809710406095b4e1a427dc0fcdc9198cd
        'random_state': [74],
        'max_iter': [5000],
        'shuffle': [False]
}
from sklearn.neural_network import MLPClassifier
gs_mlp = GridSearchCV(
	MLPClassifier(), 
	param_grid, 
	n_jobs=-1,
<<<<<<< HEAD
	cv=10).fit(X,y) # weeks for running without HPC
joblib.dump(gs_mlp, "./gridsearch/gs_mlp.pkl")
pd.DataFrame(gs_mlp.cv_results_).to_csv("./gridsearch/mlp.csv")
mlp = MLPClassifier(hidden_layer_sizes = (400, 700), activation="relu", solver="adam", alpha=0.001, learning_rate_init=0.001, random_state=74, shuffle=False, max_iter=5000).fit(X,y)
joblib.dump(mlp, "./models/bc_mlp.pkl")

# Decision Tree Classifier
# 4) 
from sklearn.tree import DecisionTreeClassifier
param_grid_dtc = {
	'criterion': ['gini', 'entropy'],
	'splitter': ['best','random'],
	'max_depth':range(2,100,1), 
	'min_samples_split':range(2,100,1), 
	'min_samples_leaf':[2], 
	'max_features':['auto', 'sqrt', 'log2'], 
	'random_state':[74]
}
gs_dtc = GridSearchCV(
	DecisionTreeClassifier(), 
	param_grid_dtc, 
	n_jobs=-1,
	cv=10).fit(X,y) 
joblib.dump(gs_dtc, "./gridsearch/gs_dtc.pkl")
pd.DataFrame(gs_dtc.cv_results_).to_csv("./gridsearch/dtc.csv")
dtc = DecisionTreeClassifier(criterion="gini",max_depth=82,max_features="log2",min_samples_leaf=2,min_samples_split=82,random_state=74,splitter="best").fit(X,y)
joblib.dump(dtc, "./models/bc_dtc.pkl")

=======
	cv=10).fit(X,y)
joblib.dump(gs_mlp, "./gridsearch/gs_mlp.pkl")
# training with best parameters
mlp = MLPClassifier(**gs_mlp.best_params_).fit(X,y)
joblib.dump(mlp, "./models/bc_mlp.pkl")

models = [firstmlp, svmrbf, lr, mlp]
>>>>>>> 04b0460809710406095b4e1a427dc0fcdc9198cd

# Training and Tuning ensembles for final selection
###################################################################
# Mixing Training Data
# Bagging
###################################################################
# evaluate bagging ensemble for classification
# Loading methods
from sklearn.ensemble import BaggingClassifier

<<<<<<< HEAD
param_grid = {
    'max_samples' : [i/100 for i in range(3,50,1)]
}

gs_bagsvm = GridSearchCV(
	BaggingClassifier(svmrbf),
	param_grid, 
	n_jobs=-1,
	cv=10).fit(X,y)
gs_baglr = GridSearchCV(
	BaggingClassifier(lr),
	param_grid, 
	n_jobs=-1,
	cv=10).fit(X,y)
gs_bagmlp = GridSearchCV(
	BaggingClassifier(mlp),
	param_grid, 
	n_jobs=-1,
	cv=10).fit(X,y)
gs_bagdtc = GridSearchCV(
	BaggingClassifier(dtc),
	param_grid, 
	n_jobs=-1,
	cv=10).fit(X,y)

joblib.dump(gs_bagsvm, "./gridsearch/gs_bagsvm.pkl")
joblib.dump(gs_baglr, "./gridsearch/gs_baglr.pkl")
joblib.dump(gs_bagmlp, "./gridsearch/gs_bagmlp.pkl")
joblib.dump(gs_bagdtc, "./gridsearch/gs_bagdtc.pkl")

pd.DataFrame(gs_bagsvm.cv_results_).to_csv("./gridsearch/bagsvm.csv")
pd.DataFrame(gs_baglr.cv_results_).to_csv("./gridsearch/baglr.csv")
pd.DataFrame(gs_bagmlp.cv_results_).to_csv("./gridsearch/bagmlp.csv")
pd.DataFrame(gs_bagdtc.cv_results_).to_csv("./gridsearch/bagdtc.csv")

bagsvm=BaggingClassifier(svmrbf, max_samples=0.43).fit(X,y)
baglr=BaggingClassifier(lr, max_samples=0.43).fit(X,y)
bagmlp=BaggingClassifier(mlp, max_samples=0.33).fit(X,y)
bagdtc=BaggingClassifier(dtc, max_samples=0.48).fit(X,y)

joblib.dump(bagsvm, "./models/bagsvm.pkl")
joblib.dump(baglr, "./models/baglr.pkl")
joblib.dump(bagmlp, "./models/bagmlp.pkl")
joblib.dump(bagdtc, "./models/bagdtc.pkl")
=======
# define the model
bagrbf = BaggingClassifier(
	svmrbf, 
	random_state=74).fit(X,y)
baglr = BaggingClassifier(
	lr, 
	random_state=74).fit(X,y)
bagmlp = BaggingClassifier(
	mlp, 
	random_state=74).fit(X,y)
bagmodels = [bagrbf, baglr, bagmlp]

# export models
joblib.dump(bagrbf, "./models/bagrbf.pkl")
joblib.dump(baglr, "./models/baglr.pkl")
joblib.dump(bagmlp, "./models/bagmlp.pkl")

>>>>>>> 04b0460809710406095b4e1a427dc0fcdc9198cd
###################################################################
# Mixing combinations of predictions
# Boosting: training over weak classifiers 
###################################################################
# Adaboost
# Loading methods for svm and lr
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
<<<<<<< HEAD
=======
from sklearn.tree import DecisionTreeClassifier
>>>>>>> 04b0460809710406095b4e1a427dc0fcdc9198cd
from sklearn.model_selection import GridSearchCV

# define the models 
adadtc = AdaBoostClassifier(
	base_estimator=DecisionTreeClassifier())
adarbf = AdaBoostClassifier(
	base_estimator=SVC(
		kernel='rbf',
		probability=True,
		random_state=74))
adalr = AdaBoostClassifier(
	base_estimator=LogisticRegression())

# Ada requires a sample weight implementation
<<<<<<< HEAD
# set parameters
param_grid_rbf = {
	'n_estimators': (10, 20, 30, 40, 50),                  
	'learning_rate': (0.001, 0.01, 0.1),
	'algorithm':['SAMME'],
	'base_estimator__C': np.logspace(-3,3,25),
	'base_estimator__gamma': np.logspace(-4,2,25)}

param_grid_lr = {
	'n_estimators': (10, 20, 30, 40, 50),                  
	'learning_rate': (0.001, 0.01, 0.1),
	'algorithm':['SAMME'],
	'base_estimator__C': np.logspace(-3,4,25),
	'base_estimator__penalty': ['l2'],
	'base_estimator__solver': ['sag'],
=======

# set parameters
param_grid_rbf = {
	'n_estimators': (1,10,25,50,100),                  
	'learning_rate': (0.0001, 0.001, 0.01, 0.1, 1.0),
	'algorithm':['SAMME'],
	'base_estimator__C': np.logspace(-3,3,10),
	'base_estimator__gamma': np.logspace(-4,2,10)}

param_grid_lr = {
	'n_estimators': (1,50,100),                  
	'learning_rate': (0.0001, 0.001, 0.01, 0.1, 1.0),
	'algorithm':['SAMME'],
	'base_estimator__C': np.logspace(-3,4,5),
	'base_estimator__penalty': ['l2'],
	'base_estimator__solver': ['newton-cg', 'lbfgs', 'sag'],
>>>>>>> 04b0460809710406095b4e1a427dc0fcdc9198cd
	'base_estimator__random_state': [74],
	'base_estimator__max_iter': [10000]}
	
param_grid_dtc = {
<<<<<<< HEAD
	'n_estimators': (10,50),                  
	'learning_rate': (0.001, 0.01, 0.1),
	'algorithm':['SAMME'],
	'base_estimator__criterion': ['gini', 'entropy'],
	'base_estimator__splitter': ['best','random'],
	'base_estimator__max_depth':range(2,100,1), 
	'base_estimator__min_samples_split':range(2,100,1), 
	'base_estimator__min_samples_leaf':[2], 
	'base_estimator__max_features':['sqrt', 'log2'], 
=======
	'n_estimators': (1,50,100),                  
	'learning_rate': (0.0001, 0.01, 0.1, 1.0),
	'algorithm':['SAMME'],
	'base_estimator__criterion': ['gini', 'entropy','log_loss'],
	'base_estimator__splitter': ['best','random'],
	'base_estimator__max_depth':range(2,100,20), 
	'base_estimator__min_samples_split':range(2,100,20), 
	'base_estimator__min_samples_leaf':range(2,100,20), 
	'base_estimator__max_features':['auto', 'sqrt', 'log2', 'None'], 
>>>>>>> 04b0460809710406095b4e1a427dc0fcdc9198cd
	'base_estimator__random_state':[74]
}

# Tuning hyperparameters 
# Try adarbf = gs_adarbf.fit(X,y)
gs_adarbf=GridSearchCV(
	adarbf,
	param_grid_rbf,
	n_jobs=-1,
	cv=10).fit(X,y) # lot of time
joblib.dump(gs_adarbf, "./gridsearch/gs_adarbf.pkl")
gs_adalr=GridSearchCV(
	adalr,
	param_grid_lr,
	n_jobs=-1,
	cv=10).fit(X,y)
joblib.dump(gs_adalr, "./gridsearch/gs_adalr.pkl")
gs_adadtc = GridSearchCV(
	adadtc,
	param_grid_dtc,
	n_jobs=-1,
	cv=10).fit(X,y)
joblib.dump(gs_adadtc, "./gridsearch/gs_adadtc.pkl")

<<<<<<< HEAD
# Export best metrics for gridsearch 

gs = [gs_adarbf, gs_adalr, gs_adadtc]

results = [pd.DataFrame(p.cv_results_) for p in gs]
Features = [list(p.columns) for p in results]
[pd.DataFrame(i,columns=j).to_csv("./gridsearch/gs_"+k+".csv",index=False) for i,j,k in zip(results,Features,["adarbf","adalr","adadtc"])]
=======
# Export best metrics for gridsearch (dtc:36000x22 + lr:1200x19 + rbf:2500x17)

gs = [gs_svmrbf,gs_lr,gs_mlp, gs_adarbf, gs_adalr, gs_adadtc]

results = [pd.DataFrame(p.cv_results_) for p in gs]
Features = [list(p.columns) for p in results]
[pd.DataFrame(i,columns=j).to_csv("./gridsearch/gs_"+k+".csv",index=False) for i,j,k in zip(results,Features,["svmrbf","lr","mlp","adarbf","adalr","adadtc"])]
>>>>>>> 04b0460809710406095b4e1a427dc0fcdc9198cd

# Train with best parameters
adarbf = AdaBoostClassifier(
	SVC(
<<<<<<< HEAD
		C= 1000,
		gamma=0.1),
	algorithm='SAMME',
	learning_rate=0.001,
	n_estimators=10).fit(X,y)
	
adalr=AdaBoostClassifier(
	LogisticRegression(
		C=10000,
		max_iter=10000,
		penalty='l2',
		random_state=74,
		solver='sag'),
	algorithm='SAMME',
	learning_rate=0.1,
	n_estimators=10).fit(X,y)

adadtc = AdaBoostClassifier(
	DecisionTreeClassifier(
		criterion='gini',
		max_depth=62,
		max_features='sqrt',
		min_samples_leaf=2,
		min_samples_split=22,
		random_state=74,
		splitter='random'),
	learning_rate=1,
=======
		C= 46.41588833612773,
		gamma=0.21544346900318823),
	algorithm='SAMME',
	learning_rate=0.1,
	n_estimators=10).fit(X,y)
adalr=AdaBoostClassifier(
	LogisticRegression(
		C=3.1622776601683795,
		max_iter=10000,
		penalty='l2',
		random_state=74,
		solver='newton-cg'),
	algorithm='SAMME',
	learning_rate=0.1,
	n_estimators=100).fit(X,y)

adadtc = AdaBoostClassifier(
	DecisionTreeClassifier(
		criterion='entropy',
		max_depth=42,
		max_features='log2',
		min_samples_leaf=2,
		min_samples_split=62,
		random_state=74,
		splitter='random'),
	learning_rate=1.0,
>>>>>>> 04b0460809710406095b4e1a427dc0fcdc9198cd
	n_estimators=100).fit(X,y)

# export model
joblib.dump(adadtc, "./models/adadtc.pkl")
joblib.dump(adalr, "./models/adalr.pkl")
joblib.dump(adarbf, "./models/adarbf.pkl")

<<<<<<< HEAD
=======
bosmodels = [adarbf, adalr, adadtc]

>>>>>>> 04b0460809710406095b4e1a427dc0fcdc9198cd
###################################################################
# Mixing models
# Voting Ensembles:
###################################################################
# Majority/Hard Voting
from sklearn.ensemble import VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
estimators = [
<<<<<<< HEAD
	('svmrbf',CalibratedClassifierCV(svmrbf).fit(X,y)),
	('lr',lr),
	('mlp',mlp),
	('dtc',dtc)]
=======
	('radial',CalibratedClassifierCV(svmrbf).fit(X,y)),
	('logistic',lr),
	('multi',mlp)]
>>>>>>> 04b0460809710406095b4e1a427dc0fcdc9198cd
hard_ensemble = VotingClassifier(
	estimators,
	voting='hard').fit(X,y)
# Implementar platt scaling para establecer learning curve en hard-ensemble_metrics
# Platt
from sklearn.linear_model import LogisticRegression
platt = pd.DataFrame(hard_ensemble.predict(Xt))
hard_ensemble = LogisticRegression().fit(platt,yt)
joblib.dump(hard_ensemble,"./models/hard_ensemble.pkl")

# Average/Soft Voting
soft_ensemble = VotingClassifier(estimators, voting='soft').fit(X,y)
joblib.dump(soft_ensemble,"./models/soft_ensemble.pkl")

# Weighted Voting
<<<<<<< HEAD
scores = [4,1,8,1]

weight_ensemble = VotingClassifier(estimators=estimators, voting='soft', weights=scores).fit(X,y)

joblib.dump(weight_ensemble,"./models/weight_ensemble.pkl")

=======

scores = [0.8936, 0.8936, 0.8723] # accuracies over weak learners

weight_ensemble = VotingClassifier(estimators=estimators, voting='soft', weights=scores).fit(X,y)
joblib.dump(weight_ensemble,"./models/weight_ensemble.pkl")

votmodels = [hard_ensemble, soft_ensemble, weight]

>>>>>>> 04b0460809710406095b4e1a427dc0fcdc9198cd
###################################################################
# Stacking: train multiple models hierarchically
###################################################################
# With sklearn

from sklearn.ensemble import StackingClassifier
<<<<<<< HEAD
estimators = [("svm", svmrbf),("mlp",mlp),("dtc",dtc)]
stack_1 = StackingClassifier(
	estimators = estimators,
	stack_method = "predict_proba",
	final_estimator = LogisticRegression()).fit(X, y)
estimators = [("svm", svmrbf),("mlp",mlp),("dtc",dtc)]
stack_2 = StackingClassifier(
	estimators = estimators,
	stack_method = "predict_proba",
=======
estimators = [("svm", svmrbf),("mlp",mlp)]
stack_1 = StackingClassifier(
	estimators = estimators,
	final_estimator = lr).fit(X, y)
estimators = [("svm", svmrbf)]
stack_2 = StackingClassifier(
	estimators = estimators,
	final_estimator = lr).fit(X, y)
estimators = [("mlp", mlp)]
stack_3 = StackingClassifier(
	estimators = estimators,
>>>>>>> 04b0460809710406095b4e1a427dc0fcdc9198cd
	final_estimator = lr).fit(X, y)

joblib.dump(stack_1, "./models/stacking_1.pkl")
joblib.dump(stack_2, "./models/stacking_2.pkl")
<<<<<<< HEAD
=======
joblib.dump(stack_3, "./models/stacking_3.pkl")

stacks = [stack_1, stack_2, stack_3]
>>>>>>> 04b0460809710406095b4e1a427dc0fcdc9198cd
