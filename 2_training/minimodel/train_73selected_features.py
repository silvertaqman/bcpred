#!/usr/bin/env python3
import numpy as np
import pandas as pd
import sklearn
import joblib
import imblearn

# Data warehousing
# Load data for BAGMLP (300 features)
bc = pd.read_csv("../../1_warehousing/Mix_BC_srbal.csv.gz")
bc_input = bc.iloc[0:466, 0:300]
bc_output = bc['Class']

# Metrics (Every model)
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, mean_squared_error, f1_score

# Data partition (mathematical notation) at 75:15:10
from sklearn.model_selection import train_test_split as tts
bcX, bcXt, bcy, bcyt = tts(
	bc_input,
	bc_output,
	random_state=74,
	test_size=0.25) # 1-trainratio

bcXv, bcXt, bcyv, bcyt = tts(
	bcXt,
	bcyt,
	random_state=74,
	test_size=0.4) #70:20:10 # testratio/(testratio+validationratio)
# Training and Tuning models

# Load data for BAGMLP (73 features)
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
	Xs,
	ys,
	random_state=74,
	test_size=0.25) # 1-trainratio

Xv, Xt, yv, yt = tts(
	Xt,
	yt,
	random_state=74,
	test_size=0.4)

# Export smoted, balanced data
X = pd.DataFrame(X, columns = sel.columns)
Xv = pd.DataFrame(Xv, columns = sel.columns)
Xt = pd.DataFrame(Xt, columns = sel.columns)

# load best model (BagMLP)
spencer = joblib.load("../models/bagmlp.pkl.gz")
#Train same models with reduced data
param_grid = {
        'hidden_layer_sizes': [x for x in itertools.product((100,80,60,40,20,15), repeat=2)],
        'activation': ['logistic', 'relu'],
        'solver': ['adam'],
        'alpha': [0.0001,0.01],
        'learning_rate_init': np.logspace(-3,-1,11),
        'random_state': [74],
        'max_iter': [5000],
        'shuffle': [False]
}
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
gs_mlp = GridSearchCV(
	MLPClassifier(), 
	param_grid, 
	n_jobs=-1,
	cv=10).fit(X,y)
joblib.dump(gs_mlp, "./gs_minimlp.pkl")
mlp = MLPClassifier(**gs_mlp.best_params_).fit(X,y)

from sklearn.ensemble import BaggingClassifier
minibagmlp = BaggingClassifier(
	mlp, 
	random_state=74).fit(X,y)

# Metrics for comparing minibag and spencer
# K-fold Validation
from sklearn.model_selection import cross_validate as cv
from sklearn.metrics import make_scorer, accuracy_score, roc_auc_score, recall_score, f1_score, log_loss, precision_score
scoring = ['accuracy','recall','precision','roc_auc','f1','neg_log_loss']
kfv_spencer = cv(spencer, bcXv, bcyv, cv=10, scoring= scoring, n_jobs=-1)
kfv_mini = cv(minibagmlp, Xv, yv, cv=10, scoring= scoring, n_jobs=-1)
kfv = [kfv_spencer, kfv_mini]

# K-stratified Validation
from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=10,shuffle=True,random_state=74)
ksfv_spencer = cv(spencer, bcXv, bcyv, cv=kfold, scoring= scoring, n_jobs=-1)
ksfv_mini = cv(minibagmlp, Xv, yv, cv=10, scoring= scoring, n_jobs=-1)
ksfv = [ksfv_spencer, ksfv_mini]
metrics = list(itertools.chain.from_iterable(zip(kfv, ksfv)))

# Exporting metrics to csv
metrics = pd.concat(map(pd.DataFrame, (metrics[i] for i in range(len(metrics)))))


metrics['folds'] = 4*['fold'+str(i+1) for i in range(10)]
modelsname = ['spencer','minibagmlp']
metrics['model'] = np.append(np.repeat(modelsname, 10),np.repeat(modelsname, 10))
metrics['method'] = np.repeat(['kfold','stratified'],160)
metrics.to_csv('./validation_metrics.csv')






