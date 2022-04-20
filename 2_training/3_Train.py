#!/usr/bin/env python3
# Preload packages
import pandas as pd
import numpy as np
import sklearn
import mglearn
import joblib

#load data
bc = pd.read_csv("Mix_BC_srbal.csv.gz")
bc_input = bc.iloc[0:466, 0:300]
bc_output = bc['Class']

# Data partition (mathematical notation)
from sklearn.model_selection import train_test_split as tts
X, Xt, y, yt = tts(bc_input,bc_output,random_state=74)

# Set Weights
from sklearn.utils import compute_class_weight as sw
weights = sw(class_weight = 'balanced', classes = np.unique(y), y = y)
cw = {i:j for i,j in zip(np.unique(y), weights)}

# Train model
## Two RBF parameters: C y gamma (anchura del nucleo)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
svm = SVC(kernel = 'rbf', C=9, gamma=0.00315).fit(X,y)
svmlin = SVC(kernel="linear",random_state=74,gamma='scale',class_weight=cw).fit(X,y)
svmrbf = SVC(kernel = 'rbf', random_state=74,gamma='scale',class_weight=cw).fit(X,y)
lr = LogisticRegression(solver='lbfgs',random_state=74,class_weight=cw).fit(X,y)
mlp = MLPClassifier(hidden_layer_sizes= (20), random_state = 74, max_iter=50000, shuffle=False).fit(X,y)

# Export models
joblib.dump(svm, "bc_svm.pkl")
joblib.dump(svmlin, "bc_svmlin.pkl")
joblib.dump(svmrbf, "bc_svmrbf.pkl")
joblib.dump(lr, "bc_lr.pkl")
joblib.dump(mlp, "bc_mlp.pkl")
