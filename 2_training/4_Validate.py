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

'''
#GridSearch
intg = [j/10000 for j in range(1,300,30)]
intc = [i/2 for i in range(18,28,1)] 
from sklearn.model_selection import GridSearchCV
param_grid = {'C': intc,'gamma': intg}
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
plt.pyplot.savefig('plot.png', dpi=300, bbox_inches='tight')
params.to_csv('params8.csv', index=False)

Problema dual: maximizas el valor de la ganancia
Los coefcientes dan los  hiperplanos que claifican entre cancer y no cancer. 
Se puede graficar el CV, como heatmap
Hacer un scatter plot con los mejores accuracy (como superficie)
'''
'''
Se buscaron los parámetros con la funcion gridsearchcv en sklearn para svm: discretiza el espacio de parametros y prueba (C: de 1 a 20) y (gamma: de 0.001 hasta 1 con salto de 0.1)

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
# Parece que existen varios maximos locales
'''
