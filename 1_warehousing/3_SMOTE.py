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
cancer = pd.read_csv("cle_bal.csv")

# SMOTE

from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=123)
Xr, yr = sm.fit_resample(Xt, yt)

# 376 × 8,744 in mix

#load dataset

from sklearn.model_selection import train_test_split
X, Xt, y, yt = train_test_split(cancer.iloc[0:377, 2:8741],cancer['V2'],random_state=0)

#split dataset: remove columns

# Train model
## Two approaches: polynomial kernel and Radial Basis Function
## Two RBF parameters: C y gamma (anchura del nucleo)

from sklearn.svm import SVC
svm = SVC(kernel = 'rbf', C=10, gamma=.1).fit(X,y)



# class labels are given by dual coefficient's sign
sv_labels = svm.dual_coef_.ravel() > 0

# C from 1e-2 to 1e5 by 10x log-scale
# gamma from 1e-5 to 10

# Scale the frequency values

#Calidad de la prediccion
svm.score(Xt, yt)
