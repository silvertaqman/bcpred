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
joblib.load("bc_svm.pkl")
joblib.load("bc_svmlin.pkl")
joblib.load("bc_svmrbf.pkl")
joblib.load("bc_lr.pkl")
joblib.load("bc_mlp.pkl")

# Usar el k-fold (k=5) y luego k-stratified (k=5) 
from sklearn.model_selection import cross_val_score as cvs
kfold_svm = cvs(svm,bc_input,bc_output,cv=5)
kfold_svmlin = cvs(svmlin,bc_input,bc_output,cv=5)
kfold_svmrbf = cvs(svmrbf,bc_input,bc_output,cv=5)
kfold_lr = cvs(lr,bc_input,bc_output,cv=5)
kfold_mlp = cvs(mlp,bc_input,bc_output,cv=5)

# K-stratified
from sklearn.model_selection import KFold
kfold = KFold(n_splits=5)
skfold_svm = cvs(svm,bc_input,bc_output,cv=kfold)
skfold_svmlin = cvs(svmlin,bc_input,bc_output,cv=kfold)
skfold_svmrbf = cvs(svmrbf,bc_input,bc_output,cv=kfold)
skfold_lr = cvs(lr,bc_input,bc_output,cv=kfold)
skfold_mlp = cvs(mlp,bc_input,bc_output,cv=kfold)

'''
GridSearch
from sklearn.model_selection import GridSearchCV
gs = GridSearchCV(SVC(), param_grid, cv=5)
gs.fit(X,y)
gs.best_estimator_
gs.best_score_
# Perfeccionar los valores
gs.best_accuracy
results = pd.DataFrame(gs.cv_results_)
Features = list(results.columns)
params = pd.DataFrame(results,columns=Features)
scores = np.array(results.mean_test_score).reshape(10, 10)
plot the mean cross-validation scores
mglearn.tools.heatmap(scores, xlabel='gamma', xticklabels=param_grid['gamma'],ylabel='C', yticklabels=param_grid['C'], cmap="viridis")

#save plots
plt.pyplot.savefig('plot8.png', dpi=300, bbox_inches='tight')
params.to_csv('params8.csv', index=False)

Problema dual: maximizas el valor de la ganancia
Los coefcientes dan los  hiperplanos que claifican entre cancer y no cancer. 
Se puede graficar el CV, como heatmap
Hacer un scatter plot con los mejores accuracy (como superficie)
'''


'''
Se buscaron los parámetros con la funcion gridsearchcv en sklearn para svm: discretiza el espacio de parametros y prueba (C: de 1 a 20) y (gamma: de 0.01 hasta 1 con salto de 0.1)

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
'''
# Metrics

from sklearn.metrics import confusion_matrix,accuracy_score,roc_auc_score,f1_score,recall_score,precision_score

# Stacking ensemble method: vecstack module
from vecstack import stacking
s, st = stacking(models, X, y, Xt, regression = True, n_folds = 4, shuffle = True, random_state = 74)
allplus = svmlin.fit(s, y)
yp = allplus.predict(st)

# Accurary and recall
accuracy_score(yp, yt)
recall_score(yp, yt)

