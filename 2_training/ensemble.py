import sys
import pandas as pd
import numpy as np
import scipy
import sklearn
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
	'n_estimators': (1,10,50,100),                  
	'learning_rate': (0.0001, 0.001, 0.01, 0.1, 1.0),
	'base_estimator__C': np.logspace(-3,4,5),
	'base_estimator__penalty': ['l2', 'none'],
	'base_estimator__solver': ['newton-cg', 'lbfgs', 'sag'],
	'base_estimator__random_state': [74],
	'base_estimator__max_iter': [1000]}

param_grid_mlp = {
	'n_estimators': (1,10,50,100),                  
	'learning_rate': (0.0001, 0.001, 0.01, 0.1, 1.0),
	'base_estimator__activation': ['tanh','logistic','relu'], 
	'base_estimator__alpha': [0.0001, 0.001, 0.05], 
	'base_estimator__hidden_layer_sizes': [(100, 15), (80,20,15), (120,80,40), (100,50,30)], 
	'base_estimator__learning_rate_init': ['constant','adaptative'], 
	'base_estimator__max_iter': [50, 100], 
	'base_estimator__random_state': 74,
	'base_estimator__shuffle': False, 
	'base_estimator__solver': ['adam','sgd']}

# Tuning hyperparameters
gs_rbf=GridSearchCV(adarbf,param_grid_rbf,n_jobs=-1,cv=5).fit(X,y) # lot of time
gs_rbf.best_params_
gs_lr=GridSearchCV(adalr,param_grid_lr,n_jobs=-1, cv=5).fit(X,y)
gs_lr.best_params_
gs_mlp=GridSearchCV(adamlp,param_grid_mlp,n_jobs=-1, cv=5).fit(X,y)
gs_mlp.best_params_

joblib.dump(gs_rbf, "./gss/gsrbf.pkl")
joblib.dump(gs_lr, "./gss/gslr.pkl")
joblib.dump(gs_mlp, "./gss/gsmlp.pkl")
