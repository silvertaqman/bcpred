#!/usr/bin/env python3
# Preload packages
import sys
import pandas as pd
import numpy as np
import matplotlib as plt
import scipy
import IPython
import sklearn
import mglearn
import joblib

# Data loading
bc = pd.read_csv("./Mix_BC_srbal.csv.gz")
bc_input = bc.iloc[0:466, 0:300]
bc_output = bc['Class']

# Load models
svm = joblib.load("bc_svm.pkl")
svmlin = joblib.load("bc_svmlin.pkl")
svmrbf = joblib.load("bc_svmrbf.pkl")
lr = joblib.load("bc_lr.pkl")
mlp = joblib.load("bc_mlp.pkl")
models = [svm, svmlin, svmrbf, lr, mlp]
# K-fold Validation
from sklearn.model_selection import cross_val_score as cvs
kfv = [cvs(p, bc_input,bc_output,cv=5) for p in models]
# K-stratified Validation
from sklearn.model_selection import KFold
kfold = KFold(n_splits=5)
ksfv = [cvs(p, bc_input,bc_output,cv=kfold) for p in models]
cols = ['fold'+str(i) for i in range(1,6)]
# Export results
metrics = pd.DataFrame(kfv+ksfv, columns = cols)
metrics['validation'] = 5*['kfold']+5*['strat']
metrics['mean_acc'] = [np.mean(i) for i in kfv+ksfv]
metrics['mean_sd'] = [np.std(i) for i in kfv+ksfv]
metrics['method'] = 2*['svm', 'svmlin', 'svmrbf', 'lr', 'mlp']
metrics.to_csv('./validation_metrics.csv')

# Metrics (Every model)
from sklearn.metrics import confusion_matrix,accuracy_score,roc_auc_score,recall_score
# Data partition (mathematical notation)
from sklearn.model_selection import train_test_split as tts
X, Xt, y, yt = tts(bc_input,bc_output,random_state=74)
# Make predictions
yp = [ svm.predict(Xt), svmlin.predict(Xt), svmrbf.predict(Xt), lr.predict(Xt), mlp.predict(Xt) ]
# Create a classification report
metrics = pd.DataFrame()
metrics['method'] = ['svm', 'svmlin', 'svmrbf', 'lr', 'mlp']
metrics['acc']=[accuracy_score(p, yt) for p in yp]
metrics['auc']=[roc_auc_score(p, yt) for p in yp]
metrics['recall']=[recall_score(p, yt) for p in yp]
metrics.to_csv('./model_metrics.csv')
