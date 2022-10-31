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

# define the models
#adarbf = AdaBoostClassifier(
#	base_estimator=SVC(
#		kernel='rbf',
#		probability=True,
#		random_state=74))
#adalr = AdaBoostClassifier(base_estimator=LogisticRegression(random_state=74))

# adamlp = AdaBoostClassifier(base_estimator=MLPClassifier(random_state=74))
# As MLP doesn't support sample_weights, method is addapted

class customMLPClassifer(MLPClassifier):
    def resample_with_replacement(self, X, y, sample_weight):
        # normalize sample_weights if not already
        sample_weight = sample_weight / sample_weight.sum(dtype=np.float64)
        X_r = np.zeros((len(X), len(X[0])), dtype=np.float64)
        y_r = np.zeros((len(y)), dtype=np.int64)
        for i in range(len(X)):
            # draw a number from 0 to len(X)-1
            draw = np.random.choice(np.arange(len(X)), p=sample_weight)
            # place the X and y at the drawn number into the resampled X_r and y_R
            X_r[i] = X[draw]
            y_r[i] = y[draw]
        return X_r, y_r
    def fit(self, X, y, sample_weight=None):
        if sample_weight is not None:
            X, y = self.resample_with_replacement(X, y, sample_weight)
        return self._fit(X, y, incremental=(self.warm_start and hasattr(self, "classes_")))

##############################################


adamlp = AdaBoostClassifier(base_estimator=MLPClassifier(random_state=74))

gs_mlp = GridSearchCV(estimator=adamlp, param_grid=param_grid_mlp,n_jobs=-1,cv=5, refit=True, verbose=1,return_train_score=False).fit(X,y)

# set parameters
#param_grid_rbf = {
	'n_estimators': (1,10,25,50,100),                  
	'learning_rate': (0.0001, 0.001, 0.01, 0.1, 1.0),
	'base_estimator__C': np.logspace(-3,3,10), # aumentar  y usar logscale 1 a 1000
	'base_estimator__gamma': np.logspace(-4,2,10)} # aumentar  y usar logscale 0.0001 hasta 10

#param_grid_lr = {
	'n_estimators': (1,10,50,100),                  
	'learning_rate': (0.0001, 0.001, 0.01, 0.1, 1.0),
	'base_estimator__C': np.logspace(-3,4,5),
	'base_estimator__penalty': ['l2', 'none'],
	'base_estimator__solver': ['newton-cg', 'lbfgs', 'sag'],
	'base_estimator__max_iter': [500, 1000]}
#If these failures are not expected, you can try to debug them by setting error_score='raise'.
	
	

# Tuning hyperparameters
#gs_rbf=GridSearchCV(adarbf,param_grid_rbf,n_jobs=-1,cv=5).fit(X,y) # lot of time
#gs_rbf.best_params_
#joblib.dump(gs_rbf, "./gss/gsrbf.pkl")
#gs_lr=GridSearchCV(adalr,param_grid_lr,n_jobs=-1, cv=5).fit(X,y)
#gs_lr.best_params_
#joblib.dump(gs_lr, "./gss/gslr.pkl")
#GridSearchCV(adamlp,param_grid_mlp,n_jobs=-1, cv=5).fit(X,y)

gs_mlp.best_params_
joblib.dump(gs_mlp, "./gss/gsmlp.pkl")
