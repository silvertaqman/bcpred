#!/usr/bin/env python3
import vecstack

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

# Metrics (Every model)
from sklearn.metrics import confusion_matrix,accuracy_score,roc_auc_score,recall_score
# Data partition (mathematical notation)
from sklearn.model_selection import train_test_split as tts
X, Xt, y, yt = tts(bc_input,bc_output,random_state=74)

# Loading models
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
svmlin = joblib.load("./models/bc_svmlin.pkl")
svmrbf = joblib.load("./models/bc_svmrbf.pkl")
lr = joblib.load("./models/bc_lr.pkl")
mlp = joblib.load("./models/bc_mlp.pkl")
models = [svmlin, svmrbf, lr, mlp]

# Mixing training data

# Bagging

### Bootstrapping, training and aggregating
### ensure independence between composition of the samples
from sklearn.ensemble import BaggingClassifier as ba
clf = [ba(base_estimator = p, n_estimators=10).fit(X, y) for p in models]

# [recall_score(p.predict(Xt),yt) for p in clf]
# Solo aumento el accuracy del svm lineal, al resto reduce o iguala
# con recall lo mismo, pero mejoro el de LR

# Mixing combinations

# Boosting: training over weak classifiers





# Stacking: train multiple models together
from vecstack import stacking
s, st = stacking(models, X, y, Xt, regression = True, n_folds = 4, shuffle = True, random_state = 74)
allplus = svmrbf.fit(s, y)
yp = allplus.predict(st)

# Mixing models

# Max/Hard Voting
from sklearn.ensemble import VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
estimators = [('linear',CalibratedClassifierCV(svmlin)),('radial',CalibratedClassifierCV(svmrbf)),('logistic',lr),('multi',mlp)]
hard_ensemble = VotingClassifier(estimators, voting='hard').fit(X,y)
hard_ensemble.score(Xt,yt)

# Average/Soft Voting

average_prediction = average_predictions = sum([p.predict(Xt) for p in models])/4.0

soft_ensemble = VotingClassifier(estimators, voting='soft').fit(X,y)
yp = soft_ensemble.predict(Xt)






