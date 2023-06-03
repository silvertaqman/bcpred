#!/usr/bin/env python3
import numpy as np
import pandas as pd
import sklearn
import joblib

#Load data
scaler = joblib.load("./1_warehousing/minmax.pkl.gz")
model = joblib.load("./2_training/models/bagsvm.pkl.gz")
descriptors = pd.read_csv("./descriptors.csv.gz")
names = pd.DataFrame(descriptors["Protein"])
descriptors = descriptors.iloc[:, 2::]

# Select features
with open("./1_warehousing/topfeatures.csv", "r") as F:
	F = F.read().split()

Xm = descriptors.filter(F[1:])
### scale
Xm = scaler.fit_transform(Xm)
Xm = pd.DataFrame(Xm, columns = F[1:])
# Predict
prob = 100*pd.DataFrame(model.predict_proba(Xm))[1]
pred = model.predict(Xm)
pred = ["Positive" if int(i) == 1 else "Negative" for i in pred]
status = pd.DataFrame({"probability":prob, "status":pred})
names = names.join(status)
print("The predicted label are:","\n", names)
