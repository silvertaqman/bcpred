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
# Gridsearch runned on HPC-Cedia cluster. Hyperparameters setted to maximize accuracy and recall responses. 

# Load data
bc = pd.read_csv("./Mix_BC_srbal.csv.gz")
bc_input = bc.iloc[0:466, 0:300]
bc_output = bc['Class']

# Metrics (Every model)
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, f1_score, log_loss

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
#######################################################################
#Arrange for hard voting
# Max/Hard Voting
from sklearn.ensemble import VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
estimators = [
	('radial',CalibratedClassifierCV(svmrbf).fit(X,y)),
	('logistic',lr),
	('multi',mlp)]
hard_ensemble = VotingClassifier(
	estimators,
	voting='hard').fit(X,y)
platt = pd.DataFrame(hard_ensemble.predict(Xt))
# load previous models (16 models)
firstmlp = joblib.load("./models/firstmlp.pkl.gz")
svmrbf = joblib.load("./models/bc_svmrbf.pkl.gz")
lr = joblib.load("./models/bc_lr.pkl.gz")
mlp = joblib.load("./models/bc_mlp.pkl.gz")
bagrbf = joblib.load("./models/bagrbf.pkl.gz")
baglr = joblib.load("./models/baglr.pkl.gz")
bagmlp = joblib.load("./models/bagmlp.pkl.gz")
adarbf = joblib.load("./models/adarbf.pkl.gz")
adalr = joblib.load("./models/adalr.pkl.gz")
adadtc = joblib.load("./models/adadtc.pkl.gz")
hard_ensemble = joblib.load("./models/hard_ensemble.pkl.gz")
soft_ensemble = joblib.load("./models/soft_ensemble.pkl.gz")
weight_ensemble = joblib.load("./models/weight_ensemble.pkl.gz")
stack_1 = joblib.load("./models/stacking_1.pkl.gz")
stack_2 = joblib.load("./models/stacking_2.pkl.gz")
stack_3 = joblib.load("./models/stacking_3.pkl.gz")
models = [firstmlp, svmrbf, lr, mlp]
bagmodels = [bagrbf, baglr, bagmlp]
bosmodels = [adarbf, adalr, adadtc]
votmodels = [hard_ensemble, soft_ensemble, weight_ensemble]
stacks = [stack_1, stack_2, stack_3]

# Prediction for ROC curves
models = models+bagmodels+bosmodels+votmodels+stacks
# Saving predicted values for cut-off evaluation (ROC curves)
yp = pd.DataFrame()
yp["Reality"]=yt
yp["firstmlp"]=np.array(pd.DataFrame(firstmlp.predict_proba(Xt))[1])
yp["svmrbf"]=svmrbf.decision_function(Xt)
yp["lr"]=lr.decision_function(Xt)
yp["mlp"]=np.array(pd.DataFrame(mlp.predict_proba(Xt))[1])
yp["bagrbf"]=bagrbf.decision_function(Xt)
yp["baglr"]=baglr.decision_function(Xt)
yp["bagmlp"]=np.array(pd.DataFrame(bagmlp.predict_proba(Xt))[1])
yp["adarbf"]=adarbf.decision_function(Xt)
yp["adalr"]=adalr.decision_function(Xt)
yp["adadtc"]=adadtc.decision_function(Xt)

yp["hard_ensemble"]=hard_ensemble.decision_function(platt)
yp["soft_ensemble"]=np.array(pd.DataFrame(soft_ensemble.predict_proba(Xt))[1])
yp["weight_ensemble"]=np.array(pd.DataFrame(weight_ensemble.predict_proba(Xt))[1])
yp["stack_1"]=stack_1.decision_function(Xt)
yp["stack_2"]=stack_2.decision_function(Xt)
yp["stack_3"]=stack_3.decision_function(Xt)

yp.to_csv("./predictions.csv")

# Validation
# K-fold Validation
from sklearn.model_selection import cross_validate as cv
from sklearn.metrics import make_scorer, accuracy_score, roc_auc_score, recall_score, f1_score, log_loss, precision_score
scoring = ['accuracy','recall','precision','roc_auc','f1','neg_log_loss']
kfv = [cv(p, Xv, yv, cv=10, scoring= scoring, n_jobs=-1) for p in models]

# K-stratified Validation
from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=10,shuffle=True,random_state=74)
ksfv = [cv(p, Xv, yv, cv=kfold, scoring= scoring, n_jobs=-1) for p in models]
metrics = list(itertools.chain.from_iterable(zip(kfv, ksfv)))

# Exporting metrics to csv
metrics = pd.concat(map(pd.DataFrame, (metrics[i] for i in range(len(metrics)))))
metrics['folds'] = 32*['fold'+str(i+1) for i in range(10)]
modelsname = ['firstmlp','svmrbf','lr', 'mlp', 'bagrbf','baglr','bagmlp','adarbf','adalr','adadtc', 'hard_ensemble','soft_ensemble','weight_ensemble','stack_1','stack_2','stack_3']
metrics['model'] = np.append(np.repeat(modelsname, 10),np.repeat(modelsname, 10))
metrics['method'] = np.repeat(['kfold','stratified'],160)
metrics.to_csv('./validation_metrics.csv')

# Learning Curve
# Export data for overfit learning curve (30x17)
from sklearn.model_selection import learning_curve
size_fmlp, score_fmlp, tscore_fmlp, ft_fmlp,_ = learning_curve(firstmlp, X, y, cv=10, train_sizes=np.linspace(0.1, 1.0, 10),return_times=True)
size_svm, score_svm, tscore_svm, ft_svm,_ = learning_curve(svmrbf, X, y, cv=10, train_sizes=np.linspace(0.1, 1.0, 10),return_times=True)
size_lr, score_lr, tscore_lr, ft_lr,_ = learning_curve(lr, X, y, cv=10, train_sizes=np.linspace(0.1, 1.0, 10),return_times=True)
size_mlp, score_mlp, tscore_mlp, ft_mlp,_ = learning_curve(mlp, X, y, cv=10, train_sizes=np.linspace(0.1, 1.0, 10),return_times=True)
size_bagsvm, score_bagsvm, tscore_bagsvm, ft_bagsvm,_ = learning_curve(bagrbf, X, y, cv=10, train_sizes=np.linspace(0.1, 1.0, 10),return_times=True)
size_baglr, score_baglr, tscore_baglr, ft_baglr,_ = learning_curve(baglr, X, y, cv=10, train_sizes=np.linspace(0.1, 1.0, 10),return_times=True)
size_bagmlp, score_bagmlp, tscore_bagmlp, ft_bagmlp,_ = learning_curve(bagmlp, X, y, cv=10, train_sizes=np.linspace(0.1, 1.0, 10),return_times=True)
size_adasvm, score_adasvm, tscore_adasvm, ft_adasvm,_ = learning_curve(adarbf, X, y, cv=10, train_sizes=np.linspace(0.1, 1.0, 10),return_times=True)
size_adalr, score_adalr, tscore_adalr, ft_adalr,_ = learning_curve(adalr, X, y, cv=10, train_sizes=np.linspace(0.1, 1.0, 10),return_times=True)
size_adadtc, score_adadtc, tscore_adadtc, ft_adadtc,_ = learning_curve(adadtc, X, y, cv=10, train_sizes=np.linspace(0.1, 1.0, 10),return_times=True)
size_hard, score_hard, tscore_hard, ft_hard,_ = learning_curve(hard_ensemble, X, y, cv=10, train_sizes=np.linspace(0.1, 1.0, 10),return_times=True)
size_soft, score_soft, tscore_soft, ft_soft,_ = learning_curve(soft_ensemble, X, y, cv=10, train_sizes=np.linspace(0.1, 1.0, 10),return_times=True)
size_weight_ensemble, score_weight_ensemble, tscore_weight_ensemble, ft_weight_ensemble,_ = learning_curve(weight_ensemble, X, y, cv=10, train_sizes=np.linspace(0.1, 1.0, 10),return_times=True)
size_stack1, score_stack1, tscore_stack1, ft_stack1,_ = learning_curve(stack_1, X, y, cv=10, train_sizes=np.linspace(0.1, 1.0, 10),return_times=True)
size_stack2, score_stack2, tscore_stack2, ft_stack2,_ = learning_curve(stack_2, X, y, cv=10, train_sizes=np.linspace(0.1, 1.0, 10),return_times=True)
size_stack3, score_stack3, tscore_stack3, ft_stack3,_ = learning_curve(stack_3, X, y, cv=10, train_sizes=np.linspace(0.1, 1.0, 10),return_times=True)

metrics = pd.DataFrame()
metrics['train_size'] = np.concatenate((size_fmlp, size_svm, size_lr, size_mlp, size_bagsvm, size_baglr, size_bagmlp, size_adasvm, size_adalr, size_adadtc, size_hard, size_soft, size_weight_ensemble, size_stack1, size_stack2, size_stack3 ))
metrics['models'] = np.repeat(modelsname, 10)
metrics = pd.concat([metrics,pd.DataFrame(np.concatenate([score_fmlp, score_svm, score_lr, score_mlp, score_bagsvm, score_baglr, score_bagmlp, score_adasvm, score_adalr, score_adadtc, score_hard, score_soft, score_weight_ensemble, score_stack1, score_stack2, score_stack3])), pd.DataFrame(np.concatenate([tscore_fmlp, tscore_svm, tscore_lr, tscore_mlp, tscore_bagsvm, tscore_baglr, tscore_bagmlp, tscore_adasvm, tscore_adalr, tscore_adadtc, tscore_hard, tscore_soft, tscore_weight_ensemble, tscore_stack1, tscore_stack2, tscore_stack3])),pd.DataFrame(np.concatenate([ft_fmlp, ft_svm, ft_lr, ft_mlp, ft_bagsvm, ft_baglr, ft_bagmlp, ft_adasvm, ft_adalr, ft_adadtc, ft_hard, ft_soft, ft_weight_ensemble, ft_stack1, ft_stack2, ft_stack3]))],axis=1)
metrics.columns = ['train_size','models']+['train_scores_fold_%d'% x for x in range(1,11)]+['validation_scores_fold_%d'% x for x in range(1,11)]+['fit_times_fold_%d'% x for x in range(1,11)]
############################################################
# Tabla comparativa para ver cual es mejor
# Comparar curvas PR por cada metodo y curvas de AUC-ROC

# Verificar una ganancia del mejor algoritmo con el algoritmo mlp original (100*(93.5 - 95.9)/93.5
# revisar las capas del mlp original y comparar 
# revisar la funcion de activacion (relu, tanh, etc)
############################################################
metrics.to_csv('./learning_curve.csv')
