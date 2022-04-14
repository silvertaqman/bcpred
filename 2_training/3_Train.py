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

#Useful Symbols: 

#save plots
#plt.pyplot.savefig('plot.png', dpi=300, bbox_inches='tight')

#load data
cancer = pd.read_csv("Mix_BC_srbal.csv.gz")

# Set Weights
from sklearn.utils import compute_class_weight 
cw = compute_class_weight(class_weight = 'balanced', classes = np.unique(y), y = y)
class_weights = {i:j for i,j in zip(np.unique(y), cw)}

# Data partition
from sklearn.model_selection import train_test_split
X, Xt, y, yt = train_test_split(cancer.iloc[0:466, 0:300],cancer['Class'],random_state=74)

# Train model
## Two approaches: polynomial kernel and Radial Basis Function
## Two RBF parameters: C y gamma (anchura del nucleo)

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
svm = SVC(kernel = 'rbf', C=9, gamma=0.00315).fit(X,y)
svmlin = SVC(kernel="linear",random_state=74,gamma='scale',class_weight=class_weights).fit(X,y)
svmrbf = SVC(kernel = 'rbf', random_state=74,gamma='scale',class_weight=class_weights).fit(X,y)
lr = LogisticRegression(solver='lbfgs',random_state=74,class_weight=class_weights).fit(X,y)
mlp = MLPClassifier(hidden_layer_sizes= (20), random_state = 74, max_iter=50000, shuffle=False).fit(X,y)
# Feature Subset selection mejora el accuracy de 0.5 a 0.769

# Se buscaron los parámetros con la funcion gridsearchcv en sklearn para svm: discretiza el espacio de parametros y prueba (C: de 1 a 20) y (gamma: de 0.01 hasta 1 con salto de 0.1)

# Grid search en busca de los parametros del SVM
# the process goes like this:
# best_score of 0.811 with C=100, gamma=0.001
# best_score of 0.796 with C=5, gamma=0.01
# best score of 0.811 with C=45, gmma = 0.001
# best_score of 0.814 with C=7, gamma=0.0044 (0.829 con Xt y yt)
# best score of 0.8137 with C=21, gamma = 0.002
# best score of 0.817 with C = 9, gamma = 0.002
# best score of 0.819 with C = 11, gamma = 0.003  (0.8205 con Xt y yt)
# best score of 0.8193374 with C = 10, gamma = 0.0028  (0.82051 con Xt y yt)
# best score of 0.817 with C = 9, gamma = 0.00315 (0.8205 con Xt y yt)
# Tratar con gamma = 0.013609
# Parece que existen varios maximos locales
#intg = [j/10000 for j in range(1,300,30)]
#intc = [i/2 for i in range(18,28,1)] 
#param_grid = {'C': intc,'gamma': intg}

# Usar el k-fold (k=5) y luego k-stratified (k=5) 
from sklearn.model_selection import cross_val_score
scores_svm = cross_val_score(svm,cancer.iloc[0:466, 0:300],cancer['Class'],cv=5)
scores_svmlin = cross_val_score(svmlin,cancer.iloc[0:466, 0:300],cancer['Class'],cv=5)
scores_svmrbf = cross_val_score(svmrbf,cancer.iloc[0:466, 0:300],cancer['Class'],cv=5)
scores_lr = cross_val_score(lr,cancer.iloc[0:466, 0:300],cancer['Class'],cv=5)
scores_mlp = cross_val_score(mlp,cancer.iloc[0:466, 0:300],cancer['Class'],cv=5)

# K-stratified
from sklearn.model_selection import KFold
kfold = KFold(n_splits=5)
scores = cross_val_score(svm,cancer.iloc[0:466, 0:300],cancer['Class'],cv=kfold)

# GridSearch
#from sklearn.model_selection import GridSearchCV
#gs = GridSearchCV(SVC(), param_grid, cv=5)
#gs.fit(X,y)

gs.best_estimator_
gs.best_score_
# Perfeccionar los valores
# gs.best_accuracy (probar)
results = pd.DataFrame(gs.cv_results_)
Features = list(results.columns)
params = pd.DataFrame(results,columns=Features)
scores = np.array(results.mean_test_score).reshape(10, 10)
# plot the mean cross-validation scores
# mglearn.tools.heatmap(scores, xlabel='gamma', xticklabels=param_grid['gamma'],ylabel='C', yticklabels=param_grid['C'], cmap="viridis")
# plt.pyplot.savefig('plot8.png')
#params.to_csv('params8.csv', index=False)
#Problema dual: maximizas el valor de la ganancia
# Los coefcientes dan los  hiperplanos que claifican entre cancer y no cancer. 

# Se puede graficar el CV, como heatmap
# Hacer un scatter plot con los mejores accuracy (como superficie)

#Metrics
from sklearn.metrics import confusion_matrix,accuracy_score, roc_auc_score,f1_score, recall_score, precision_score
'''
# SVM linear, SVM, LR and MLP (from author).
SVC(kernel=\"linear\",random_state=seed,gamma='scale',class_weight=class_weights)
SVC(kernel = 'rbf', random_state=seed,gamma='scale',class_weight=class_weights),\n
LogisticRegression(solver='lbfgs',random_state=seed,class_weight=class_weights)
MLPClassifier(hidden_layer_sizes= (20), random_state = seed, max_iter=50000, shuffle=False)
'''
