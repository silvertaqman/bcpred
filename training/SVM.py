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
plt.pyplot.savefig('plot.png', dpi=300, bbox_inches='tight')

#load data
cancer = pd.read_csv("")

#37 rows in AC
# 8709 x 373 in mix

#split datasets

from sklearn.model_selection import train_test_split
X, Xt, y, yt = train_test_split(cancer[['A','R','N','D','C','E','Q','G','H','I','L','L','K','M','F','P','S','T','S','T','W','Y','V']],cancer['V2'],random_state=0)

#Use too: cancer.iloc[0, 2:25]

# Train model
## Two approaches: polynomial kernel and Radial Basis Function
## Two RBF parameters: C y gamma (anchura del nucleo)

from sklearn.svm import SVC
svm = SVC(kernel = 'rbf', C=10, gamma=.1).fit(X,y)
cvgridsearch
# class labels are given by dual coefficient's sign
sv_labels = svm.dual_coef_.ravel() > 0

# C from 1e-2 to 1e5 by 10x log-scale
# gamma from 1e-5 to 10

# Scale the frequency values

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
cancer = pd.DataFrame(cancer.data, columns=cancer.feature_names)
cancer.to_csv('cancer.csv')

np.mean(yt)
