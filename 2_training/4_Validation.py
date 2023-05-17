#!/usr/bin/env python3
###################################################################
# Preload packages
###################################################################
import sys
import pandas as pd
import numpy as np
import sklearn
import joblib
import itertools
# Gridsearch runned on HPC-Cedia cluster. Hyperparameters setted to maximize accuracy and recall responses. 

# Load data
bc = pd.read_csv("./Mix_BC_srbal.csv.gz")
bc_input = bc.iloc[0:466, 0:350]
bc_output = bc['Class']

# Metrics (Every model)
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, f1_score
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
# load previous models (16 models)
svmrbf = joblib.load("./models/bc_svmrbf.pkl.gz")
lr = joblib.load("./models/bc_lr.pkl.gz")
mlp = joblib.load("./models/bc_mlp.pkl.gz")
dtc = joblib.load("./models/bc_dtc.pkl.gz")
# Max/Hard Voting
from sklearn.ensemble import VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
estimators = [
	('radial',svmrbf),
	('logistic',lr),
	('multi',mlp),
	('dtc',dtc)]
hard_ensemble = VotingClassifier(
	estimators,
	voting='hard').fit(X,y)
platt = pd.DataFrame(hard_ensemble.predict(Xt))
# Predictions
M1 = joblib.load("./models/hard_ensemble.pkl.gz")
M2 = joblib.load("./models/soft_ensemble.pkl.gz")
M3 = joblib.load("./models/weight_ensemble.pkl.gz")

M4 = joblib.load("./models/stacking_1.pkl.gz")
M5 = joblib.load("./models/stacking_2.pkl.gz")

M6 = joblib.load("./models/bagrbf.pkl.gz")
M7 = joblib.load("./models/baglr.pkl.gz")
M8 = joblib.load("./models/bagmlp.pkl.gz")
M9 = joblib.load("./models/bagdtc.pkl.gz")

M10 = joblib.load("./models/adarbf.pkl.gz")
M11 = joblib.load("./models/adalr.pkl.gz")
M12 = joblib.load("./models/adadtc.pkl.gz")


models = [svmrbf, lr, mlp, dtc]
bagmodels = [M6, M7, M8, M9]
bosmodels = [M10, M11, M12]
votmodels = [M1,M2,M3]
stacks = [M4, M5]

# Prediction for ROC curves
models = models+bagmodels+bosmodels+votmodels+stacks
# Saving predicted values for cut-off evaluation (ROC curves)
yp = pd.DataFrame()
yp["Reality"]=yt
yp["svmrbf"]=svmrbf.fit(X,y).decision_function(Xt)
yp["lr"]=lr.decision_function(Xt)
yp["mlp"]=np.array(pd.DataFrame(mlp.predict_proba(Xt))[1])
yp["dtc"]=np.array(pd.DataFrame(mlp.predict_proba(Xt))[1])

yp["M1"]=M1.decision_function(platt)
yp["M2"]=np.array(pd.DataFrame(M2.predict_proba(Xt))[1])
yp["M3"]=np.array(pd.DataFrame(M3.predict_proba(Xt))[1])
yp["M4"]=M4.decision_function(Xt)
yp["M5"]=np.array(pd.DataFrame(M5.predict_proba(Xt))[1])

yp["M6"]=M6.decision_function(Xt)
yp["M7"]=M7.decision_function(Xt)
yp["M8"]=np.array(pd.DataFrame(M8.predict_proba(Xt))[1])
yp["M9"]=np.array(pd.DataFrame(M9.predict_proba(Xt))[1])

yp["M10"]=M10.decision_function(Xt)
yp["M11"]=M11.decision_function(Xt)
yp["M12"]=M12.decision_function(Xt)

yp.to_csv("./predictions.csv")

# Validation
# K-fold Validation
from sklearn.model_selection import cross_validate as cv
from sklearn.metrics import make_scorer, accuracy_score, roc_auc_score, recall_score, f1_score, precision_score
scoring = ['accuracy','recall','precision','roc_auc','f1']
kfv = [cv(p, bc_input, bc_output, cv=10, scoring= scoring, n_jobs=-1) for p in models]

# K-stratified Validation
from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=10,shuffle=True,random_state=74)
ksfv = [cv(p, bc_input, bc_output, cv=kfold, scoring= scoring, n_jobs=-1) for p in models]
metrics = list(itertools.chain.from_iterable(zip(kfv, ksfv)))

# Exporting metrics to csv
metrics = pd.concat(map(pd.DataFrame, (metrics[i] for i in range(len(metrics)))))
metrics['folds'] = 32*['fold'+str(i+1) for i in range(10)]
modelsname = ['svmrbf','lr', 'mlp','dtc', 'M6','M7','M8','M9','M10','M11','M12', 'M1','M2','M3','M4','M5']
metrics['model'] = np.append(np.repeat(modelsname, 10),np.repeat(modelsname, 10))
metrics['method'] = np.repeat(['kfold','stratified'],160)
metrics.to_csv('./validation_metrics.csv')

# Learning Curve
# Export data for overfit learning curve (30x17)
from sklearn.model_selection import learning_curve
size_svm, score_svm, tscore_svm, ft_svm,_ = learning_curve(svmrbf, X, y, cv=10, train_sizes=np.linspace(0.1, 1.0, 10),return_times=True)
size_lr, score_lr, tscore_lr, ft_lr,_ = learning_curve(lr, X, y, cv=10, train_sizes=np.linspace(0.1, 1.0, 10),return_times=True)
size_mlp, score_mlp, tscore_mlp, ft_mlp,_ = learning_curve(mlp, X, y, cv=10, train_sizes=np.linspace(0.1, 1.0, 10),return_times=True)
size_dtc, score_dtc, tscore_dtc, ft_dtc,_ = learning_curve(dtc, X, y, cv=10, train_sizes=np.linspace(0.1, 1.0, 10),return_times=True)
size_M6, score_M6, tscore_M6, ft_M6,_ = learning_curve(M6, X, y, cv=10, train_sizes=np.linspace(0.1, 1.0, 10),return_times=True)
size_M7, score_M7, tscore_M7, ft_M7,_ = learning_curve(M7, X, y, cv=10, train_sizes=np.linspace(0.1, 1.0, 10),return_times=True)
size_M8, score_M8, tscore_M8, ft_M8,_ = learning_curve(M8, X, y, cv=10, train_sizes=np.linspace(0.1, 1.0, 10),return_times=True)
size_M9, score_M9, tscore_M9, ft_M9,_ = learning_curve(M9, X, y, cv=10, train_sizes=np.linspace(0.1, 1.0, 10),return_times=True)
size_M10, score_M10, tscore_M10, ft_M10,_ = learning_curve(M10, X, y, cv=10, train_sizes=np.linspace(0.1, 1.0, 10),return_times=True)
size_M11, score_M11, tscore_M11, ft_M11,_ = learning_curve(M11, X, y, cv=10, train_sizes=np.linspace(0.1, 1.0, 10),return_times=True)
size_M12, score_M12, tscore_M12, ft_M12,_ = learning_curve(M12, X, y, cv=10, train_sizes=np.linspace(0.1, 1.0, 10),return_times=True)
size_M1, score_M1, tscore_M1, ft_M1,_ = learning_curve(M1, X, y, cv=10, train_sizes=np.linspace(0.1, 1.0, 10),return_times=True)
size_M2, score_M2, tscore_M2, ft_M2,_ = learning_curve(M2, X, y, cv=10, train_sizes=np.linspace(0.1, 1.0, 10),return_times=True)
size_M3, score_M3, tscore_M3, ft_M3,_ = learning_curve(M3, X, y, cv=10, train_sizes=np.linspace(0.1, 1.0, 10),return_times=True)
size_M4, score_M4, tscore_M4, ft_M4,_ = learning_curve(M4, X, y, cv=10, train_sizes=np.linspace(0.1, 1.0, 10),return_times=True)
size_M5, score_M5, tscore_M5, ft_M5,_ = learning_curve(M5, X, y, cv=10, train_sizes=np.linspace(0.1, 1.0, 10),return_times=True)

metrics = pd.DataFrame()
metrics['train_size'] = np.concatenate((size_svm, size_lr, size_mlp, size_dtc, size_M6, size_M7, size_M8, size_M9, size_M10, size_M11, size_M12, size_M1, size_M2, size_M3, size_M4, size_M5))
metrics['models'] = np.repeat(modelsname, 10)
metrics = pd.concat(
[metrics,pd.DataFrame(np.concatenate((score_svm, score_lr, score_mlp, score_M6, score_M7, score_M8, score_M9, score_M10, score_M11, score_M12, score_M1, score_M2, score_M3, score_M4, score_M5))), pd.DataFrame(np.concatenate((tscore_svm, tscore_lr, tscore_mlp, tscore_M6, tscore_M7, tscore_M8, tscore_M9, tscore_M10, tscore_M11, tscore_M12, tscore_M1, tscore_M2, tscore_M3, tscore_M4, score_M5))),pd.DataFrame(np.concatenate((ft_svm, ft_lr, ft_mlp, ft_M6, ft_M7, ft_M8, ft_M9, ft_M10, ft_M11, ft_M12, ft_M1, ft_M2, ft_M3, ft_M4, score_M5)))],axis=1)
metrics.columns = ['train_size','models']+['train_scores_fold_%d'% x for x in range(1,11)]+['validation_scores_fold_%d'% x for x in range(1,11)]+['fit_times_fold_%d'% x for x in range(1,11)]
############################################################
# Tabla comparativa para ver cual es mejor
# Comparar curvas PR por cada metodo y curvas de AUC-ROC

# Verificar una ganancia del mejor algoritmo con el algoritmo mlp original (100*(93.5 - 95.9)/93.5
# revisar las capas del mlp original y comparar 
# revisar la funcion de activacion (relu, tanh, etc)
############################################################
metrics.to_csv('./learning_curve.csv')
