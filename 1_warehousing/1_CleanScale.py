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
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNetCV

# Compare methods: kbest, percentile, False positive rate, false discovery rate, family wise, ridgeCV and Tree Model based
from sklearn.ensemble import ExtraTreesClassifier
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

# RidgeCV
clf = RidgeCV(alphas=np.logspace(-5,5,10)).fit(X,y)
importance = np.abs(clf.coef_)
threshold = np.sort(importance)[-300] + 0.01
ridge = SelectFromModel(clf, threshold=threshold).fit(X, y)

# LassoCV
clf = LassoCV(alphas=np.logspace(-5,5,100)).fit(X,y)
importance = np.abs(clf.coef_)
threshold = np.sort(importance)[-300] + 0.01
lasso = SelectFromModel(clf, threshold=threshold).fit(X, y)

# ElasticNetCV
clf = ElasticNetCV(l1_ratio = [0.05, 0.1, 0.5, 0.9, 0.95],alphas=np.logspace(-5,5,200), cv=10).fit(X,y)
importance = np.abs(clf.coef_)
threshold = np.sort(importance)[-300] + 0.01
elasticnet = SelectFromModel(clf, threshold=threshold).fit(X, y)

# Final comparison

sel=[p.fit_transform(X,y) for p in selector.values()]
features = [p.get_support(indices=True) for p in selector.values()]
features.append(ridge.get_support(indices=True))
features.append(lasso.get_support(indices=True))
features.append(elasticnet.get_support(indices=True))
total_feat=[]
for i in features:
	total_feat.append(np.array([F[k] for k in i]))


selector['ridgeCV']=[]
selector['lassoCV']=[]
selector['ElasticNetCV']=[]
metrics = {k:f for k,f in zip(selector.keys(),total_feat)}
metrics = pd.DataFrame().from_dict(metrics, orient="index").T
metrics.to_csv("selection.csv")

# Compare sel features trough dendrograms/heatmap (graphic)

# Using kbest Chi2
selector = SelectKBest(chi2, k=300)
X = selector.fit_transform(X, y)

### export unselected features
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
