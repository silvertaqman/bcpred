#!/usr/bin/env python3
import numpy as np
import pandas as pd
import sklearn
import joblib
<<<<<<< HEAD
import sys

#Load data
scaler = joblib.load("../1_warehousing/minmax.pkl.gz")
model = joblib.load(sys.argv[1])
=======

#Load data
scaler = joblib.load("../1_warehousing/minmax.pkl")
bagmlp = joblib.load("../2_training/models/bagmlp.pkl.gz")
stack1 = joblib.load("../2_training/models/stacking_1.pkl.gz")
>>>>>>> 04b0460809710406095b4e1a427dc0fcdc9198cd

metastasis = pd.read_csv("./Screening_1_Metastasis.csv.gz")
immuno = pd.read_csv("./Screening_2_Cancer_Immunotherapy_Genes.csv.gz")
rbps = pd.read_csv("./Screening_3_RBPs.csv.gz")

## Mathematical notation
Xm = metastasis.drop('Class', axis = 1)
Xi = immuno.drop('Class', axis = 1)
Xr = rbps.drop('Class', axis = 1)
# Select features
<<<<<<< HEAD
with open("../1_warehousing/topfeatures.csv", "r") as F:
=======
with open("../2_training/Selected_Features.txt", "r") as F:
>>>>>>> 04b0460809710406095b4e1a427dc0fcdc9198cd
	F = F.read().split()

Xm = Xm.filter(F)
Xi = Xi.filter(F)
Xr = Xr.filter(F)
### scale
Xm = scaler.fit_transform(Xm)
<<<<<<< HEAD
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
=======
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
>>>>>>> 04b0460809710406095b4e1a427dc0fcdc9198cd

X = [Xm, Xi, Xr]
y = [ym, yi, yr]

<<<<<<< HEAD
# model acc is higher
mae=[mean_absolute_error(model.predict(X[i]),y[i]) for i in range(3)]
print(pd.DataFrame([acc,mae]))
=======
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
>>>>>>> 04b0460809710406095b4e1a427dc0fcdc9198cd
