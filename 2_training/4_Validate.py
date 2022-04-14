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
cancer = pd.read_csv("Mix_BreastCancer_srbal.csv.gz")

from sklearn.model_selection import train_test_split
X, Xt, y, yt = train_test_split(cancer.iloc[0:466, 2:8708],cancer['Class'],random_state=0)

#split dataset: remove columns

# Train model
## Two approaches: polynomial kernel and Radial Basis Function
## Two RBF parameters: C y gamma (anchura del nucleo)

from sklearn.svm import SVC
svm = SVC(kernel = 'rbf', C=10, gamma=.1).fit(X,y)
#Calidad de la prediccion
svm.score(Xt, yt)

# class labels are given by dual coefficient's sign
sv_labels = svm.dual_coef_.ravel() > 0

# Se buscaron los parámetros con la funcion gridsearchcv en sklearn: discretiza el espacio de parametros y prueba (C: de 1 a 20) y (gamma: de 0.01 hasta 1 con salto de 0.1)

# Grid search en busca de los parametros del SVM
# intg = [i/100 for i in range(1, 110, 20)]
# intc = [j for j in range(1, 22, 4)]
intg = [i/100 for i in range(1, 110, 20)] #(0.001 hasta 2)
intc = [j for j in range(1, 22, 4)] #(20 hasta 100)
param_grid = {'C': intc,'gamma': intg}

# Usar el k-fold (k=5) y luego k-stratified (k=5) 

# Luego esto
from sklearn.model_selection import GridSearchCV
gs = GridSearchCV(SVC(), param_grid, cv=3)
gs.fit(X,y)

gs.best_estimator_
# Perfeccionar los valores
# gs.best_accuracy (probar)
# SVC(C=1.01, gamma=1) para g (0.01 a 1) y c(1, 21)
results = pd.DataFrame(gs.cv_results_)
Features = list(results.columns)
params = pd.DataFrame(results,columns=Features)
params.to_csv('params.csv', index=False)

#Problema dual: maximizas el valor de la ganancia
# Los coefcientes dan los  hiperplanos que claifican entre cancer y no cancer. 

# Se puede graficar el CV, como heatmap. 
# Hacer un scatter plot con los mejores accuracy (como superficie)
