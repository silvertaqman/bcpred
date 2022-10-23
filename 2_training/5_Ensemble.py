#!/usr/bin/env python3
###################################################################
# Preload packages
###################################################################
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import IPython
import sklearn
import mglearn
import joblib
import itertools

# Data loading
######################################################################################################################################
bc = pd.read_csv("./Mix_BC_srbal.csv.gz")
bc_input = bc.iloc[0:466, 0:300]
bc_output = bc['Class']

# Metrics (Every model)
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, mean_squared_error, log_loss

# Data partition (mathematical notation)
from sklearn.model_selection import train_test_split as tts
X, Xt, y, yt = tts(bc_input,bc_output,random_state=74)

# Loading models
#from sklearn.svm import SVC
#from sklearn.neural_network import MLPClassifier
#from sklearn.linear_model import LogisticRegression
svmrbf = joblib.load("./models/bc_svmrbf.pkl")
lr = joblib.load("./models/bc_lr.pkl")
mlp = joblib.load("./models/bc_mlp.pkl")
models = [svmrbf, lr, mlp]

# Training and Tuning ensembles for final selection
###################################################################
# Mixing Training Data
# Bagging
###################################################################
# evaluate bagging ensemble for classification
# Loading methods
from sklearn.model_selection import cross_validate as cv
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import BaggingClassifier

# define the model
bagrbf = BaggingClassifier(svmrbf, random_state=74).fit(X,y)
baglr = BaggingClassifier(svmrbf, random_state=74).fit(X,y)
bagmlp = BaggingClassifier(svmrbf, random_state=74).fit(X,y)
bagmodels = [bagrbf, baglr, bagmlp]

# export models
joblib.dump(bagrbf, "./ensemble_models/bagrbf.pkl")
joblib.dump(baglr, "./ensemble_models/baglr.pkl")
joblib.dump(bagmlp, "./ensemble_models/bagmlp.pkl")

# K-fold Validation
kfcv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=74)
scoring = ['accuracy','recall','precision','roc_auc']
kfv = [cv(p, bc_input, bc_output, scoring=scoring, cv=kfcv, n_jobs=-1, error_score='raise') for p in bagmodels]

# K-stratified Validation
ksfcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=74)
ksfv = [cv(p, bc_input, bc_output, scoring=scoring, cv=ksfcv, n_jobs=-1, error_score='raise') for p in bagmodels]
metrics = list(itertools.chain.from_iterable(zip(kfv, ksfv)))

# Exporting metrics to csv
metrics = pd.concat(map(pd.DataFrame, (metrics[i] for i in range(len(metrics)))))
metrics['repeat'] = 30*['fold'+str(i+1) for i in range(3)]
metrics['folds'] = 18*['fold'+str(i+1) for i in range(5)]
model = np.repeat(['svmrbf', 'lr', 'mlp'], 5)
metrics['model'] = np.tile(model, 6)
metrics['method'] = np.repeat(['kfold','stratified'],45)
metrics.to_csv('./ensemble_metrics/bagging_validation_metrics.csv')

# Export data for overfit learning curve
from sklearn.model_selection import learning_curve
size_svm, score_svm, tscore_svm, ft_svm,_ = learning_curve(bagrbf, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10),return_times=True)
size_lr, score_lr, tscore_lr, ft_lr,_ = learning_curve(baglr, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10),return_times=True)
size_mlp, score_mlp, tscore_mlp, ft_mlp,_ = learning_curve(bagmlp, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10),return_times=True)

metrics = pd.DataFrame()
metrics['train_size'] = np.concatenate((size_svm, size_lr, size_mlp))
metrics['models'] = 10*["svm"]+10*['lr']+10*['mlp']
metrics = pd.concat([metrics,pd.DataFrame(np.concatenate([score_svm, score_lr, score_mlp])), pd.DataFrame(np.concatenate([tscore_svm, tscore_lr, tscore_mlp])),pd.DataFrame(np.concatenate([ft_svm, ft_lr, ft_mlp]))],axis=1)
metrics.columns = ['train_size','models','train_scores_fold1','train_scores_fold2','train_scores_fold3','train_scores_fold4','train_scores_fold5','test_scores_fold1','test_scores_fold2','test_scores_fold3','test_scores_fold4','test_scores_fold5','fit_times_fold1','fit_times_fold2','fit_times_fold3','fit_times_fold4','fit_times_fold5']
metrics.to_csv('./ensemble_metrics/bagging_learning_curve.csv')
###################################################################
# Mixing combinations of predictions
# Boosting: training over weak classifiers
###################################################################
# Adaboost
# Loading methods
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_validate as cv
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

model = AdaBoostClassifier(
    base_estimator=LogisticRegression(penalty='l2', C=100, max_iter=1),
    n_estimators=4,
    random_state=42,
    )


# define the models
adarbf = AdaBoostClassifier(
	base_estimator=SVC(
		kernel='rbf',
		probability=True,
		random_state=74))
adalr = AdaBoostClassifier(base_estimator=LogisticRegression())
adamlp = AdaBoostClassifier(base_estimator=MLPClassifier())

# set parameters
param_grid_rbf = {
	'n_estimators': (1,10,25,50,100),                  
	'learning_rate': (0.0001, 0.001, 0.01, 0.1, 1.0),
	'base_estimator__C': np.logspace(-3,3,10), # aumentar  y usar logscale 1 a 1000
	'base_estimator__gamma': np.logspace(-4,2,10)} # aumentar  y usar logscale 0.0001 hasta 10

param_grid_lr = {
	'n_estimators': (1,50,100),                  
	'learning_rate': (0.0001, 0.001, 0.01, 0.1, 1.0),
	'base_estimator__C': np.logspace(-3,4,5),
	'base_estimator__penalty': ['l2', 'none'],
	'base_estimator__solver': ['newton-cg', 'lbfgs', 'sag'],
	'base_estimator__random_state': [74],
	'base_estimator__max_iter': [1000]}

param_grid_mlp = {
	'activation': ['tanh','logistic','relu'], 
	'alpha': [0.0001, 0.001, 0.05], 
	'hidden_layer_sizes': [(100, 15), (80,20,15), (120,80,40), (100,50,30)], 
	'learning_rate_init': ['constant','adaptative'], 
	'max_iter': [50, 100], 
	'random_state': 74,
	'shuffle': False, 
	'solver': ['adam','sgd']}

# Tuning hyperparameters
gs_rbf=GridSearchCV(adarbf,param_grid_rbf,n_jobs=-1,cv=5).fit(X,y) # lot of time
gs_rbf.best_params_
gs_lr=GridSearchCV(adalr,param_grid_lr,n_jobs=-1, cv=5).fit(X,y)
gs_lr.best_params_
gs_mlp=GridSearchCV(adamlp,param_grid_mlp,n_jobs=-1, cv=5).fit(X,y)
gs_mlp.best_params_

joblib.dump(gs_rbf, "./gsrbf.pkl")
joblib.dump(gs_lr, "./gslr.pkl")
joblib.dump(gs_mlp, "./gsmlp.pkl")

# Train with best parameters
adarbf = AdaBoostClassifier(base_estimator=SVC(kernel="rbf",probability=True, random_state=74, C=1, gamma=0.1), learning_rate=0.1, n_estimators=100).fit(X,y)

adalr=AdaBoostClassifier(base_estimator=LogisticRegression(penalty = "none", solver = "sag", max_iter=5000), learning_rate=0.1, n_estimators=100).fit(X,y)


# export model
joblib.dump(adarbf, "./ensemble_metrics/adamlp.pkl")
joblib.dump(adalr, "./ensemble_metrics/adalr.pkl")
joblib.dump(adamlp, "./ensemble_metrics/adarbf.pkl")


# Compare LR and SVC clasiffiers with ADA plus boosting

# K-fold Validation
kfcv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=74)
scoring = ['accuracy','recall','precision','roc_auc']
kfv = [cv(p, bc_input, bc_output, scoring=scoring, cv=kfcv, n_jobs=-1, error_score='raise') for p in bagmodels]

# K-stratified Validation
ksfcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=74)
ksfv = [cv(p, bc_input, bc_output, scoring=scoring, cv=ksfcv, n_jobs=-1, error_score='raise') for p in bagmodels]
metrics = list(itertools.chain.from_iterable(zip(kfv, ksfv)))

# Exporting metrics to csv
metrics = pd.concat(map(pd.DataFrame, (metrics[i] for i in range(len(metrics)))))
metrics['repeat'] = 30*['fold'+str(i+1) for i in range(3)]
metrics['folds'] = 18*['fold'+str(i+1) for i in range(5)]
model = np.repeat(['svmrbf', 'lr', 'mlp'], 5)
metrics['model'] = np.tile(model, 6)
metrics['method'] = np.repeat(['kfold','stratified'],45)
metrics.to_csv('./ensemble_metrics/bagging_validation_metrics.csv')

# Export data for overfit learning curve
from sklearn.model_selection import learning_curve
size_svm, score_svm, tscore_svm, ft_svm,_ = learning_curve(bagrbf, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10),return_times=True)
size_lr, score_lr, tscore_lr, ft_lr,_ = learning_curve(baglr, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10),return_times=True)
size_mlp, score_mlp, tscore_mlp, ft_mlp,_ = learning_curve(bagmlp, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10),return_times=True)

metrics = pd.DataFrame()
metrics['train_size'] = np.concatenate((size_svm, size_lr, size_mlp))
metrics['models'] = 10*["svm"]+10*['lr']+10*['mlp']
metrics = pd.concat([metrics,pd.DataFrame(np.concatenate([score_svm, score_lr, score_mlp])), pd.DataFrame(np.concatenate([tscore_svm, tscore_lr, tscore_mlp])),pd.DataFrame(np.concatenate([ft_svm, ft_lr, ft_mlp]))],axis=1)
metrics.columns = ['train_size','models','train_scores_fold1','train_scores_fold2','train_scores_fold3','train_scores_fold4','train_scores_fold5','test_scores_fold1','test_scores_fold2','test_scores_fold3','test_scores_fold4','test_scores_fold5','fit_times_fold1','fit_times_fold2','fit_times_fold3','fit_times_fold4','fit_times_fold5']
metrics.to_csv('./ensemble_metrics/bagging_learning_curve.csv')

# K-fold Validation
kfcv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=74)
scoring = ['accuracy','recall','precision','roc_auc']
kfv = [cv(p, bc_input, bc_output, scoring=scoring, cv=kfcv, n_jobs=-1, error_score='raise') for p in bagmodels]

# K-stratified Validation
ksfcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=74)
ksfv = [cv(p, bc_input, bc_output, scoring=scoring, cv=ksfcv, n_jobs=-1, error_score='raise') for p in bagmodels]
metrics = list(itertools.chain.from_iterable(zip(kfv, ksfv)))

# Exporting metrics to csv
metrics = pd.concat(map(pd.DataFrame, (metrics[i] for i in range(len(metrics)))))
metrics['repeat'] = 30*['fold'+str(i+1) for i in range(3)]
metrics['folds'] = 18*['fold'+str(i+1) for i in range(5)]
model = np.repeat(['svmrbf', 'lr', 'mlp'], 5)
metrics['model'] = np.tile(model, 6)
metrics['method'] = np.repeat(['kfold','stratified'],45)
metrics.to_csv('./ensemble_metrics/bagging_validation_metrics.csv')

# Export data for overfit learning curve
from sklearn.model_selection import learning_curve
size_svm, score_svm, tscore_svm, ft_svm,_ = learning_curve(bagrbf, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10),return_times=True)
size_lr, score_lr, tscore_lr, ft_lr,_ = learning_curve(baglr, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10),return_times=True)
size_mlp, score_mlp, tscore_mlp, ft_mlp,_ = learning_curve(bagmlp, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10),return_times=True)

metrics = pd.DataFrame()
metrics['train_size'] = np.concatenate((size_svm, size_lr, size_mlp))
metrics['models'] = 10*["svm"]+10*['lr']+10*['mlp']
metrics = pd.concat([metrics,pd.DataFrame(np.concatenate([score_svm, score_lr, score_mlp])), pd.DataFrame(np.concatenate([tscore_svm, tscore_lr, tscore_mlp])),pd.DataFrame(np.concatenate([ft_svm, ft_lr, ft_mlp]))],axis=1)
metrics.columns = ['train_size','models','train_scores_fold1','train_scores_fold2','train_scores_fold3','train_scores_fold4','train_scores_fold5','test_scores_fold1','test_scores_fold2','test_scores_fold3','test_scores_fold4','test_scores_fold5','fit_times_fold1','fit_times_fold2','fit_times_fold3','fit_times_fold4','fit_times_fold5']
metrics.to_csv('./ensemble_metrics/bagging_learning_curve.csv')

# Rotation Forest

import rotation_forest

###################################################################
# Mixing models
# Voting Ensembles:
###################################################################
# Max/Hard Voting
from sklearn.ensemble import VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
estimators = [('radial',CalibratedClassifierCV(svmrbf)),('logistic',lr),('multi',mlp)]
hard_ensemble = VotingClassifier(estimators, voting='hard').fit(X,y)
hard_ensemble.score(Xt,yt)

# K-fold Validation
# K-stratified Validation
# Exporting metrics to csv
# Export data for overfit learning curve

# Average/Soft Voting
soft_ensemble = VotingClassifier(estimators, voting='soft').fit(X,y)
soft_ensemble.score(Xt, yt)

# K-fold Validation
# K-stratified Validation
# Exporting metrics to csv
# Export data for overfit learning curve

# Hyperparameter Tuning Ensembles Over MLP
from sklearn.neural_network import MLPClassifier
mlp_1 = MLPClassifier(activation="relu", alpha=0.0001, hidden_layer=(80,20), learning_rate_init=0.001, max_iter=50000, random_state=74, shuffle=False, solver="adam")
mlp_2 = MLPClassifier(activation="relu", alpha=0.0001, hidden_layer=(20,15), learning_rate_init=0.002, max_iter=50000, random_state=74, shuffle=False, solver="adam")
mlp_3 = MLPClassifier(activation="relu", alpha=0.0001, hidden_layer=(20, 15), learning_rate_init=0.01, max_iter=50000, random_state=74, shuffle=False, solver="adam")
estimators = [('mlp_1', mlp_1), ('mlp_2', mlp_2), ('mlp_3', mlp_3)]
hte = VotingClassifier(estimators, voting='hard').fit(X,y)
hte.score(Xt,yt)

# K-fold Validation
# K-stratified Validation
# Exporting metrics to csv
# Export data for overfit learning curve

# Horizontal Voting Ensembles & Snapshot ensembles are options for future applications
# printing log loss between actual and predicted value

print("log_loss: ", log_loss(yt, yp))

###################################################################
# Stacking: train multiple models together
###################################################################
# With sklearn

from sklearn.ensemble import StackingClassifier

estimators = [("svm", svmrbf),("lr",lr)]

stack = StackingClassifier(estimators = estimators, final_estimator = mlp).fit(X, y)

joblib.dump(stack, "./ensemble_metrics/stacking.pkl")

# K-fold Validation
# K-stratified Validation
# Exporting metrics to csv
# Export data for overfit learning curve

# With mlens

from mlens.ensemble import SuperLearner

ensemble = SuperLearner(scorer = accuracy_score,random_state = 74)
ensemble.add([svmrbf, mlp])
ensemble.add_meta(lr)
ensemble.fit(X,y)

# Buscar un nuevo discriminador de caracteristicas, diferente de SelectKBest(Chi2)
# Usar funciones equivalentes / tema de combinatoria 
# Hasta comparar el discriminador y los modelos de clasificacion. 
# RandomSearch
# Intentar kfold con 10
