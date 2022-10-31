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
baglr = BaggingClassifier(lr, random_state=74).fit(X,y)
bagmlp = BaggingClassifier(mlp, random_state=74).fit(X,y)
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

# Exporting metrics to csv (90x10)
metrics = pd.concat(map(pd.DataFrame, (metrics[i] for i in range(len(metrics)))))
metrics['repeat'] = 30*['fold'+str(i+1) for i in range(3)]
metrics['folds'] = 18*['fold'+str(i+1) for i in range(5)]
model = np.repeat(['svmrbf', 'lr', 'mlp'], 5)
metrics['model'] = np.tile(model, 6)
metrics['method'] = np.repeat(['kfold','stratified'],45)
metrics.to_csv('./ensemble_metrics/bagging_validation_metrics.csv')

# Export data for overfit learning curve (30x17)
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
# Loading methods for svm and lr
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_validate as cv
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

# define the models
adadtc = AdaBoostClassifier(base_estimator=DecisionTreeClassifier())
adarbf = AdaBoostClassifier(base_estimator=SVC(kernel='rbf',probability=True,random_state=74))
adalr = AdaBoostClassifier(base_estimator=())

# Ada requires a sample weight implementation

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
	
param_grid_dtc = {
	'n_estimators': (1,50,100),                  
	'learning_rate': (0.0001, 0.01, 0.1, 1.0),
	'base_estimator__criterion': ['gini', 'entropy','log_loss'],
	'base_estimator__splitter': ['best','random'],
	'base_estimator__max_depth':range(2,100,20), 
	'base_estimator__min_samples_split':range(2,100,20), 
	'base_estimator__min_samples_leaf':range(2,100,20), 
	'base_estimator__max_features':['auto', 'sqrt', 'log2', 'None'], 
	'base_estimator__random_state':[74]
}

# Tuning hyperparameters
gs_rbf=GridSearchCV(adarbf,param_grid_rbf,n_jobs=-1,cv=5).fit(X,y) # lot of time
gs_rbf.best_params_
joblib.dump(gs_rbf, "./gsrbf.pkl")
gs_lr=GridSearchCV(adalr,param_grid_lr,n_jobs=-1, cv=5).fit(X,y)
gs_lr.best_params_
joblib.dump(gs_lr, "./gslr.pkl")
gs_dtc = GridSearchCV(adadtc,param_grid_dtc,n_jobs=-1, cv=5).fit(X,y)
gs_dtc.best_params_
joblib.dump(gs_dtc, "./gss/gsdtc.pkl")

# Export best metrics for gridsearch (dtc:36000x22 + lr:1200x19 + rbf:2500x17)


gs = [gs_dtc, gs_rbf, gs_lr]

results = [pd.DataFrame(p.cv_results_) for p in gs]
Features = [list(p.columns) for p in results]
[pd.DataFrame(i,columns=j).to_csv("./gss/"+k+".csv", index=False) for i,j,k in zip(results, Features,["dtc","rbf","lr"])]

# Train with best parameters
adarbf = AdaBoostClassifier(base_estimator=SVC(kernel="rbf",probability=True, random_state=74, C=215.44346900318823, gamma=0.21544346900318823), learning_rate=0.1, n_estimators=10).fit(X,y)
adalr=AdaBoostClassifier(base_estimator=LogisticRegression(C=10000.0,max_iter=500,penalty="l2",solver="newton-cg"),learning_rate=0.0001, n_estimators=1).fit(X,y)
adadtc = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(criterion='entropy',max_depth=82,max_features='sqrt',min_samples_leaf=1,min_samples_split=22,random_state=74,splitter='random'),learning_rate=1.0,n_estimators=100).fit(X,y)

# export model
joblib.dump(adadtc, "./ensemble_metrics/adadtc.pkl")
joblib.dump(adalr, "./ensemble_metrics/adalr.pkl")
joblib.dump(adarbf, "./ensemble_metrics/adarbf.pkl")

bosmodels = [adarbf, adalr, adadtc]

# Metrics for validation

# K-fold Validation
kfcv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=74)
scoring = ['accuracy','recall','precision','roc_auc']
kfv = [cv(p, bc_input, bc_output, scoring=scoring, cv=kfcv, n_jobs=-1, error_score='raise') for p in bosmodels]

# K-stratified Validation
ksfcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=74)
ksfv = [cv(p, bc_input, bc_output, scoring=scoring, cv=ksfcv, n_jobs=-1, error_score='raise') for p in bosmodels]
metrics = list(itertools.chain.from_iterable(zip(kfv, ksfv)))

# Exporting metrics to csv (90x10)
metrics = pd.concat(map(pd.DataFrame, (metrics[i] for i in range(len(metrics)))))
metrics['repeat'] = 30*['fold'+str(i+1) for i in range(3)]
metrics['folds'] = 18*['fold'+str(i+1) for i in range(5)]
model = np.repeat(['rbf', 'lr', 'mlp'], 5)
metrics['model'] = np.tile(model, 6)
metrics['method'] = np.repeat(['kfold','stratified'],45)
metrics.to_csv('./ensemble_metrics/boosting_validation_metrics.csv')

# Export data for overfit learning curve (30x17)
from sklearn.model_selection import learning_curve
size_svm, score_svm, tscore_svm, ft_svm,_ = learning_curve(adarbf, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10),return_times=True)
size_lr, score_lr, tscore_lr, ft_lr,_ = learning_curve(adalr, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10),return_times=True)
size_dtc, score_dtc, tscore_dtc, ft_dtc,_ = learning_curve(adadtc, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10),return_times=True)

metrics = pd.DataFrame()
metrics['train_size'] = np.concatenate((size_svm, size_lr, size_dtc))
metrics['models'] = 10*["svm"]+10*['lr']+10*['dtc']
metrics = pd.concat([metrics,pd.DataFrame(np.concatenate([score_svm, score_lr, score_dtc])), pd.DataFrame(np.concatenate([tscore_svm, tscore_lr, tscore_dtc])),pd.DataFrame(np.concatenate([ft_svm, ft_lr, ft_dtc]))],axis=1)
metrics.columns = ['train_size','models','train_scores_fold1','train_scores_fold2','train_scores_fold3','train_scores_fold4','train_scores_fold5','test_scores_fold1','test_scores_fold2','test_scores_fold3','test_scores_fold4','test_scores_fold5','fit_times_fold1','fit_times_fold2','fit_times_fold3','fit_times_fold4','fit_times_fold5']
metrics.to_csv('./ensemble_metrics/boosting_learning_curve.csv')

###################################################################
# Mixing models
# Voting Ensembles:
###################################################################
# Max/Hard Voting
from sklearn.ensemble import VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
estimators = [('radial',CalibratedClassifierCV(svmrbf).fit(X,y)),('logistic',lr),('multi',mlp)]
hard_ensemble = VotingClassifier(estimators, voting='hard').fit(X,y)
# Implementar platt scaling para establecer learning curve en hard-ensemble_metrics
# Platt
from sklearn.linear_model import LogisticRegression
platt = pd.DataFrame(hard_ensemble.predict(Xt))
hard_ensemble = LogisticRegression().fit(platt,yt)
joblib.dump(hard_ensemble,"./ensemble_models/hard_ensemble.pkl")

# Average/Soft Voting
soft_ensemble = VotingClassifier(estimators, voting='soft').fit(X,y)
soft_ensemble.score(Xt, yt)
joblib.dump(soft_ensemble,"./ensemble_models/soft_ensemble.pkl")

# Hyperparameter Tuning Ensembles Over MLP (params from previous gs)
from sklearn.neural_network import MLPClassifier
mlp_1 = MLPClassifier(activation="relu", alpha=0.0001, hidden_layer_sizes=(80,20), learning_rate_init=0.001, max_iter=50000, random_state=74, shuffle=False, solver="adam")
mlp_2 = MLPClassifier(activation="relu", alpha=0.0001, hidden_layer_sizes=(20,15), learning_rate_init=0.002, max_iter=50000, random_state=74, shuffle=False, solver="adam")
mlp_3 = MLPClassifier(activation="relu", alpha=0.0001, hidden_layer_sizes=(20, 15), learning_rate_init=0.01, max_iter=50000, random_state=74, shuffle=False, solver="adam")
estimators = [('mlp_1', mlp_1), ('mlp_2', mlp_2), ('mlp_3', mlp_3)]
hte = VotingClassifier(estimators, voting='soft').fit(X,y)
hte.score(Xt,yt)
joblib.dump(hte,"./ensemble_models/hte.pkl")

votmodels = [hard_ensemble, soft_ensemble, hte]

# K-fold Validation
kfcv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=74)
scoring = ['accuracy','recall','precision','roc_auc']
kfv = [cv(p, bc_input, bc_output, scoring=scoring, cv=kfcv, n_jobs=-1, error_score='raise') for p in votmodels]

# K-stratified Validation
ksfcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=74)
ksfv = [cv(p, bc_input, bc_output, scoring=scoring, cv=ksfcv, n_jobs=-1, error_score='raise') for p in votmodels]
metrics = list(itertools.chain.from_iterable(zip(kfv, ksfv)))

# Exporting metrics to csv (90x10)
metrics = pd.concat(map(pd.DataFrame, (metrics[i] for i in range(len(metrics)))))
metrics['repeat'] = 30*['fold'+str(i+1) for i in range(3)]
metrics['folds'] = 18*['fold'+str(i+1) for i in range(5)]
model = np.repeat(['hard', 'soft', 'hte'], 5)
metrics['model'] = np.tile(model, 6)
metrics['method'] = np.repeat(['kfold','stratified'],45)
metrics.to_csv('./ensemble_metrics/voting_validation_metrics.csv')

# Export data for overfit learning curve
from sklearn.model_selection import learning_curve
size_hard, score_hard, tscore_hard, ft_hard,_ = learning_curve(hard_ensemble, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10),return_times=True)
size_soft, score_soft, tscore_soft, ft_soft,_ = learning_curve(soft_ensemble, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10),return_times=True)
size_hte, score_hte, tscore_hte, ft_hte,_ = learning_curve(hte, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10),return_times=True)

metrics = pd.DataFrame()
metrics['train_size'] = np.concatenate((size_hard, size_soft, size_hte))
metrics['models'] = 10*["hard"]+10*['soft']+10*['hte']
metrics = pd.concat([metrics,pd.DataFrame(np.concatenate([score_hard, score_soft, score_hte])), pd.DataFrame(np.concatenate([tscore_hard, tscore_soft, tscore_hte])),pd.DataFrame(np.concatenate([ft_hard, ft_soft, ft_hte]))],axis=1)
metrics.columns = ['train_size','models','train_scores_fold1','train_scores_fold2','train_scores_fold3','train_scores_fold4','train_scores_fold5','test_scores_fold1','test_scores_fold2','test_scores_fold3','test_scores_fold4','test_scores_fold5','fit_times_fold1','fit_times_fold2','fit_times_fold3','fit_times_fold4','fit_times_fold5']
metrics.to_csv('./ensemble_metrics/voting_learning_curve.csv')

# printing log loss between actual and predicted value
#print("log_loss: ", log_loss(yt, yp))

###################################################################
# Stacking: train multiple models together
###################################################################
# With sklearn

from sklearn.ensemble import StackingClassifier
estimators = [("svm", svmrbf),("mlp",mlp)]
stack_1 = StackingClassifier(estimators = estimators, final_estimator = lr).fit(X, y)
estimators = [("hte", hte),("dtc", adadtc)]
stack_2 = StackingClassifier(estimators = estimators, final_estimator = adarbf).fit(X, y)
estimators = [("rbf", adarbf),("soft_ensemble", soft_ensemble)]
stack_3 = StackingClassifier(estimators = estimators, final_estimator = adalr).fit(X, y)

joblib.dump(stack_1, "./ensemble_models/stacking_1.pkl")
joblib.dump(stack_2, "./ensemble_models/stacking_2.pkl")
joblib.dump(stack_3, "./ensemble_models/stacking_3.pkl")

stacks = [stack_1, stack_2, stack_3]
# load previous models
#bagrbf = joblib.load("./ensemble_models/bagrbf.pkl.gz")
#baglr = joblib.load("./ensemble_models/baglr.pkl.gz")
#bagmlp = joblib.load("./ensemble_models/bagmlp.pkl.gz")
#adarbf = joblib.load("./ensemble_models/adarbf.pkl.gz")
#adalr = joblib.load("./ensemble_models/adalr.pkl.gz")
#adadtc = joblib.load("./ensemble_models/adadtc.pkl.gz")

# K-fold Validation
kfcv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=74)
scoring = ['accuracy','recall','precision','roc_auc']
kfv = [cv(p, bc_input, bc_output, scoring=scoring, cv=kfcv, n_jobs=-1, error_score='raise') for p in stacks]

# K-stratified Validation
ksfcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=74)
ksfv = [cv(p, bc_input, bc_output, scoring=scoring, cv=ksfcv, n_jobs=-1, error_score='raise') for p in stacks]
metrics = list(itertools.chain.from_iterable(zip(kfv, ksfv)))

# Exporting metrics to csv
metrics = pd.concat(map(pd.DataFrame, (metrics[i] for i in range(len(metrics)))))
metrics['repeat'] = 30*['fold'+str(i+1) for i in range(3)]
metrics['folds'] = 18*['fold'+str(i+1) for i in range(5)]
model = np.repeat(['stack_1', 'stack_2', 'stack_3'], 5)
metrics['model'] = np.tile(model, 6)
metrics['method'] = np.repeat(['kfold','stratified'],45)
metrics.to_csv('./ensemble_metrics/stacking_validation_metrics.csv')

# Export data for overfit learning curve (30x17)
from sklearn.model_selection import learning_curve
size_1, score_1, tscore_1, ft_1,_ = learning_curve(stack_1, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10),return_times=True)
size_2, score_2, tscore_2, ft_2,_ = learning_curve(stack_2, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10),return_times=True)
size_3, score_3, tscore_3, ft_3,_ = learning_curve(stack_3, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10),return_times=True)

metrics = pd.DataFrame()
metrics['train_size'] = np.concatenate((size_hard, size_soft, size_hte))
metrics['models'] = 10*["stack_1"]+10*['stack_2']+10*['stack_3']
metrics = pd.concat([metrics,pd.DataFrame(np.concatenate([score_1, score_2, score_3])), pd.DataFrame(np.concatenate([tscore_1, tscore_2, tscore_3])),pd.DataFrame(np.concatenate([ft_1, ft_2, ft_3]))],axis=1)
metrics.columns = ['train_size','models','train_scores_fold1','train_scores_fold2','train_scores_fold3','train_scores_fold4','train_scores_fold5','test_scores_fold1','test_scores_fold2','test_scores_fold3','test_scores_fold4','test_scores_fold5','fit_times_fold1','fit_times_fold2','fit_times_fold3','fit_times_fold4','fit_times_fold5']
metrics.to_csv('./ensemble_metrics/stacking_learning_curve.csv')

# Buscar un nuevo discriminador de caracteristicas, diferente de SelectKBest(Chi2)
# Usar funciones equivalentes / tema de combinatoria 
# Hasta comparar el discriminador y los modelos de clasificacion. 
# RandomSearch
# Intentar kfold con 10
