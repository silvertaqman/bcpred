#!/usr/bin/env python3
import numpy as np
import pandas as pd
import sklearn
import joblib
import sys

#Load data
scaler = joblib.load("../1_warehousing/minmax.pkl.gz")
model = joblib.load(sys.argv[1])

metastasis = pd.read_csv("./Screening_1_Metastasis.csv.gz")
immuno = pd.read_csv("./Screening_2_Cancer_Immunotherapy_Genes.csv.gz")
rbps = pd.read_csv("./Screening_3_RBPs.csv.gz")

## Mathematical notation
Xm = metastasis.drop('Class', axis = 1)
Xi = immuno.drop('Class', axis = 1)
Xr = rbps.drop('Class', axis = 1)
# Select features
with open("../1_warehousing/topfeatures.csv", "r") as F:
	F = F.read().split()

Xm = Xm.filter(F)
Xi = Xi.filter(F)
Xr = Xr.filter(F)
### scale
Xm = scaler.fit_transform(Xm)
Xm = pd.DataFrame(Xm, columns = F[1:])
Xi = scaler.fit_transform(Xi)
Xi = pd.DataFrame(Xi, columns = F[1:])
Xr = scaler.fit_transform(Xr)
Xr = pd.DataFrame(Xr, columns = F[1:])
# Predict (All cases are positive)
ym = [0]*1903
yi = [0]*1232
yr = [0]*1369
# Comparison (between stack1 and model)
from sklearn.metrics import accuracy_score, f1_score, log_loss, mean_absolute_error

X = [Xm, Xi, Xr]
y = [ym, yi, yr]

acc=[accuracy_score(model.predict(X[i]),y[i]) for i in range(3)]
# model acc is higher
mae=[mean_absolute_error(model.predict(X[i]),y[i]) for i in range(3)]
print(pd.DataFrame([acc,mae]))
