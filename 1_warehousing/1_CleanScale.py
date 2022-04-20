#!/usr/bin/env python3
import numpy as np
import pandas as pd
import sklearn
import joblib

#Load data
cancer = pd.read_csv("./Mix_BC.csv.gz")
#Data Warehousing and Feature selection
## Change name of ID and class columns
cancer.columns = list(cancer.columns[:-2])+["ProtID","Class"]
## Reordering IDs and Class columns
cancer.insert(0, 'ProtID', cancer.pop('ProtID'))

## MinMax Scale to numerical data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(cancer.iloc[0:377, 3:8744])
scaled = scaler.fit_transform(cancer.iloc[0:377, 3:8744])
cancer_s = pd.DataFrame(scaled, columns = cancer.iloc[0:377, 3:8744].columns)
### exporting scaler: loadable with joblib.load("minmax.pkl"")
joblib.dump(scaler, "minmax.pkl") 

## Remove duplicated values (No duplicates)
cancer_s.drop_duplicates(keep=False, inplace=True)

## Mathematical notation
y = cancer_s['Class']
X = cancer_s.drop('Class', axis = 1)
F = list(X.columns)

## Remove invariant columns 
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2
selector= VarianceThreshold()
X = selector.fit_transform(X.values)
### export invariant features
FS = []
[FS.append(F[i]) for i in selector.get_support(indices=True)]
with open("invariant_features.txt","w") as f:
	[f.write("%s\n" % i) for i in list(set(F) - set(FS))]

F = FS

## Feature Subset Selection
selector = SelectKBest(chi2, k=300)
X = selector.fit_transform(X, y)
### export unselected features
FS = []
[FS.append(F[i]) for i in selector.get_support(indices=True)]
with open("univariant_feature_selection.txt","w") as f:
	[f.write("%s\n" % i) for i in list(set(F) - set(FS))]

## Create and export the working dataframe: scaled and reduced
cancer_sr = pd.DataFrame(X, columns=FS)
cancer_sr['Class'] = y
cancer_sr.to_csv("./Mix_BC_sr.csv", index=False)
'''
## Principal Component Analysis (from 8708 to 332 to explain 0.97 of variance)
from sklearn.decomposition import PCA
pcaModel = PCA(0.97)
'''
