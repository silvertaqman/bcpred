#!/usr/bin/env python3
import numpy as np
import pandas as pd
import sklearn
import joblib

#Load data
<<<<<<< HEAD
scaler = joblib.load("./1_warehousing/minmax.pkl.gz")
model = joblib.load("./2_training/models/bagsvm.pkl.gz")
=======
scaler = joblib.load("./1_warehousing/minmax.pkl")
bagmlp = joblib.load("./2_training/models/bagmlp.pkl.gz")
>>>>>>> 04b0460809710406095b4e1a427dc0fcdc9198cd
descriptors = pd.read_csv("./descriptors.csv.gz")
names = pd.DataFrame(descriptors["Protein"])
descriptors = descriptors.iloc[:, 2::]

# Select features
<<<<<<< HEAD
with open("./1_warehousing/topfeatures.csv", "r") as F:
	F = F.read().split()

Xm = descriptors.filter(F[1:])
### scale
Xm = scaler.fit_transform(Xm)
Xm = pd.DataFrame(Xm, columns = F[1:])
# Predict
prob = 100*pd.DataFrame(model.predict_proba(Xm))[1]
pred = model.predict(Xm)
=======
with open("./2_training/Selected_Features.txt", "r") as F:
	F = F.read().split()

Xm = descriptors.filter(F)
### scale
Xm = scaler.fit_transform(Xm)
Xm = pd.DataFrame(Xm, columns = F)
# Predict
prob = 100*pd.DataFrame(bagmlp.predict_proba(Xm))[1]
pred = bagmlp.predict(Xm)
>>>>>>> 04b0460809710406095b4e1a427dc0fcdc9198cd
pred = ["Positive" if int(i) == 1 else "Negative" for i in pred]
status = pd.DataFrame({"probability":prob, "status":pred})
names = names.join(status)
print("The predicted label are:","\n", names)
