#!/usr/bin/env python3
import numpy as np
import pandas as pd
import sklearn
import joblib

#Load data
scaler = joblib.load("../1_warehousing/minmax.pkl")
bagmlp = joblib.load("../2_training/models/bagmlp.pkl.gz")
stack1 = joblib.load("../2_training/models/stacking_1.pkl.gz")

metastasis = pd.read_csv("./Screening_1_Metastasis.csv.gz")
immuno = pd.read_csv("./Screening_2_Cancer_Immunotherapy_Genes.csv.gz")
rbps = pd.read_csv("./Screening_3_RBPs.csv.gz")

## Mathematical notation
Xm = metastasis.drop('Class', axis = 1)
Xi = immuno.drop('Class', axis = 1)
Xr = rbps.drop('Class', axis = 1)
# Select features
with open("../2_training/Selected_Features.txt", "r") as F:
	F = F.read().split()

Xm = Xm.filter(F)
Xi = Xi.filter(F)
Xr = Xr.filter(F)
### scale
Xm = scaler.fit_transform(Xm)
Xm = pd.DataFrame(Xm, columns = F)
Xi = scaler.fit_transform(Xi)
Xi = pd.DataFrame(Xi, columns = F)
Xr = scaler.fit_transform(Xr)
Xr = pd.DataFrame(Xr, columns = F)
# Predict (All cases are positive)
ym = [1]*1903
yi = [1]*1232
yr = [1]*1369
# Comparison (between stack1 and bagmlp)
from sklearn.metrics import accuracy_score, f1_score, log_loss

X = [Xm, Xi, Xr]
y = [ym, yi, yr]

[accuracy_score(stack1.predict(X[i]),y[i]) for i in range(3)]
[accuracy_score(bagmlp.predict(X[i]),y[i]) for i in range(3)]
# bagmlp acc is higher than stack_1
[f1_score(stack1.predict(X[i]),y[i]) for i in range(3)]
[f1_score(bagmlp.predict(X[i]),y[i]) for i in range(3)]
# bagmlp f1 is higher than stack_1
[log_loss(stack1.predict(X[i]),y[i]) for i in range(3)]
[log_loss(bagmlp.predict(X[i]),y[i]) for i in range(3)]
# bagmlp log_loss is lower than stack_1
# Bagmlp is selected
