#!/usr/bin/env python3
# Preload packages
import pandas as pd
import numpy as np
import sklearn
import joblib
import itertools
# Gridsearch runned on HPC-Cedia cluster. Hyperparameters setted to maximize accuracy and recall responses. 

# Load data
bc = pd.read_csv("./Mix_BC_srbal.csv.gz")
bc_input = bc.iloc[0:466, 0:300]
bc_output = bc['Class']

# Data partition (mathematical notation)
from sklearn.model_selection import train_test_split as tts
X, Xt, y, yt = tts(bc_input,bc_output,random_state=74) #75:25

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

# Logistic regression
# 2) generates an accuracy of 0.91 with C: 3, gamma= 0.8
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
# 3) 

'''
hidden_layer = [x for x in itertools.product((128*4,128*3,128*2,128*1,128/2,128/4,128/8), repeat=2)]  # repeat indica el numero de capas.
Cambiar el 128, buscar la curva de sobreajuste para los 4. 
Agregar desviacion estandar
'''

param_grid = {
        'hidden_layer_sizes': [x for x in itertools.product((100,80,60,40,20,15), repeat=2)],
        'activation': ['logistic', 'relu'],
        'solver': ['adam'],
        'alpha': [0.0001],
        'learning_rate_init': np.logspace(-3,-1,11),
        'random_state': [74],
        'max_iter': [50000],
        'shuffle': [False]
}
from sklearn.neural_network import MLPClassifier as MLPC
gs = GridSearchCV(MLPC(), param_grid, cv=5).fit(X,y)
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
models = [svmrbf, lr, mlp]
# Saving predicted values for cut-off evaluation (ROC curves)
yp = pd.DataFrame()
yp["Reality"]=bc_output
yp["svmrbf"]=svmrbf.decision_function(bc_input)
yp["lr"]=lr.decision_function(bc_input)
yp["mlp"]=pd.DataFrame(mlp.predict_proba(bc_input))[1]
yp.to_csv("./predictions.csv")

# K-fold Validation
from sklearn.model_selection import cross_validate as cv
from sklearn.metrics import make_scorer, accuracy_score, recall_score, roc_auc_score, confusion_matrix, precision_score
scoring = ['accuracy','recall','precision','roc_auc']
kfv = [cv(p, bc_input,bc_output,cv=5,scoring= scoring, n_jobs=-1) for p in models]

# K-stratified Validation
from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=74)
ksfv = [cv(p, bc_input,bc_output,cv=kfold,scoring= scoring, n_jobs=-1) for p in models]
metrics = list(itertools.chain.from_iterable(zip(kfv, ksfv)))

# Exporting metrics to csv
metrics = pd.concat(map(pd.DataFrame, (metrics[i] for i in range(len(metrics)))))
metrics['folds'] = 6*['fold'+str(i+1) for i in range(5)]
models = ['svmrbf', 'lr', 'mlp']
metrics['model'] = np.append(np.repeat(models, 5),np.repeat(models, 5))
metrics['method'] = np.repeat(['kfold','stratified'],15)
metrics.to_csv('./validation_metrics.csv')
'''
# Export confusion matrices to csv
pd.DataFrame(confusion_matrix(yt, svmrbf.predict(Xt), labels=[0,1])).to_csv('./svmrbf_cm.csv')
pd.DataFrame(confusion_matrix(yt, lr.predict(Xt), labels=[0,1])).to_csv('./lr_cm.csv')
pd.DataFrame(confusion_matrix(yt, mlp.predict(Xt), labels=[0,1])).to_csv('./mlp_cm.csv')
'''

# Export data for overfit learning curve
from sklearn.model_selection import learning_curve
size_svm, score_svm, tscore_svm, ft_svm,_ = learning_curve(svmrbf, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10),return_times=True)
size_lr, score_lr, tscore_lr, ft_lr,_ = learning_curve(lr, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10),return_times=True)
size_mlp, score_mlp, tscore_mlp, ft_mlp,_ = learning_curve(mlp, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10),return_times=True)

metrics = pd.DataFrame()
metrics['train_size'] = np.concatenate((size_svm, size_lr, size_mlp))
metrics['models'] = 10*["svm"]+10*['lr']+10*['mlp']
metrics = pd.concat([metrics,pd.DataFrame(np.concatenate([score_svm, score_lr, score_mlp])), pd.DataFrame(np.concatenate([tscore_svm, tscore_lr, tscore_mlp])),pd.DataFrame(np.concatenate([ft_svm, ft_lr, ft_mlp]))],axis=1)
metrics.columns = ['train_size','models','train_scores_fold1','train_scores_fold2','train_scores_fold3','train_scores_fold4','train_scores_fold5','test_scores_fold1','test_scores_fold2','test_scores_fold3','test_scores_fold4','test_scores_fold5','fit_times_fold1','fit_times_fold2','fit_times_fold3','fit_times_fold4','fit_times_fold5']
metrics.to_csv('./learning_curve.csv')

'''
# Export data for overfit validation curve
from sklearn.model_selection import validation_curve
param_range = np.logspace(-3,-1,11)
train_scores, valid_scores = validation_curve(svmrbf, X, y,param_name="gamma", param_range=param_range, cv=5)
'''
