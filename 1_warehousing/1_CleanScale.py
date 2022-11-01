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

from sklearn.feature_selection import VarianceThreshold, SelectKBest, SelectFromModel, SelectPercentile, SelectFpr, SelectFdr, SelectFwe, chi2, f_classif, mutual_info_classif
## Remove invariant columns 
selector= VarianceThreshold()
X = selector.fit_transform(X.values)

### export invariant features
FS = []
[FS.append(F[i]) for i in selector.get_support(indices=True)]
with open("invariant_features.txt","w") as f:
	[f.write("%s\n" % i) for i in list(set(F) - set(FS))]

F = FS

## Feature Subset Selection 
from sklearn.linear_model import RidgeCV
# Compare methods: kbest, percentile, False positive rate, false discovery rate, family wise, ridgeCV and Tree Model based
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
selector = {
	'kbest_chi2': SelectKBest(chi2, k=300),
	'kbest_f':SelectKBest(f_classif), 
	'kbest_mutual':SelectKBest(mutual_info_classif),
	'perc_chi2':SelectPercentile(chi2, percentile=3),
	'perc_f':SelectPercentile(percentile=3),
	'perc_mutual':SelectPercentile(mutual_info_classif, percentile=3),
	'fpr_chi2':SelectFpr(chi2, alpha=1e-1),
	'fpr_f':SelectFpr(f_classif, alpha=0.015),
#	'fpr_mutual':SelectFpr(mutual_info_classif(X,y), alpha=1e-1),
#	'fdr_chi2':SelectFdr(chi2, alpha=1e-1),
	'fdr_f':SelectFdr(f_classif, alpha=1e-1),
#	'fdr_mutual':SelectFdr(mutual_info_classif, alpha=1e-1),
#	'fwe_chi2':SelectFwe(chi2, alpha=1e-1),
	'fwe_f':SelectFwe(f_classif, alpha=1e-1)
#	'fwe_mutual':SelectFwe(mutual_info_classif, alpha=1e-1)
}
sel=[p.fit_transform(X,y) for p in selector.values()]
features = [p.get_support(indices=True) for p in selector.values()]
metrics = {k:f for k,f in zip(selector.keys(),features)}
metrics = pd.DataFrame().from_dict(metrics, orient="index").T
metrics.to_csv("selection.csv")

# Compare sel features trough 

# Escoger 300 con RidgeCV
importance = np.abs(ridge.coef_)
feature_names = np.array(diabetes.feature_names)
threshold = np.sort(importance)[-300] + 0.01
sfm = SelectFromModel(ridge, threshold=threshold).fit(X, y)
print(f"Features selected by SelectFromModel: {feature_names[sfm.get_support()]}")
# tree
('tree_model',ExtraTreesClassifier(n_estimators=300))
clf = ExtraTreesClassifier(n_estimators=50)
X_tree = SelectFromModel(clf, prefit=True).transform(X)

# Using kbest (as authors)
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
## Export IDs and Class
bc[["ProtID", "Class"]].to_csv("./ProtIDs.csv")
'''
