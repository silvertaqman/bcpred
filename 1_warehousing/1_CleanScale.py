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

from sklearn.feature_selection import VarianceThreshold, SelectKBest, SelectFromModel, SelectPercentile, chi2, f_classif, mutual_info_classif
## Remove invariant columns 
selector= VarianceThreshold()
Xs = selector.fit_transform(X.values)

### export invariant features
FS = []
[FS.append(F[i]) for i in selector.get_support(indices=True)]
with open("invariant_features.txt","w") as f:
	[f.write("%s\n" % i) for i in list(set(F) - set(FS))]

F = FS

## Feature Subset Selection 

## Principal Component Analysis 
Xs = pd.DataFrame(Xs, columns = FS)
from sklearn.decomposition import PCA
pca = PCA(0.99).fit(Xs.T)
CP = pd.DataFrame(pca.components_.T)
CP.to_csv("PCAComponents.csv")
EV = pd.DataFrame(pca.explained_variance_ratio_.T)
EV.to_csv("PCAVarianceRatios.csv")

## We can explain 99 percent of variance with nearly 350 feats
## Now we use FSS with chi2, Anova-F and mutual-info 

from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNetCV

# Compare methods: k-top ordered, p-top ordered and regularization

from sklearn.ensemble import ExtraTreesClassifier
selector = {
	'kbest_chi2': SelectKBest(chi2, k=350),
	'kbest_f':SelectKBest(f_classif, k=350), 
	'kbest_mutual':SelectKBest(mutual_info_classif, k=350),
	'perc_chi2':SelectPercentile(chi2, percentile=3),
	'perc_f':SelectPercentile(percentile=3.5),
	'perc_mutual':SelectPercentile(mutual_info_classif, percentile=3.5)
}

# ElasticNetCV
clf = ElasticNetCV(l1_ratio = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95], alphas=np.logspace(-5,5,200), cv=10).fit(Xs,y)
importance = np.abs(clf.coef_)
threshold = np.sort(importance)[-350] + 0.01
elasticnet = SelectFromModel(clf, threshold=threshold).fit(Xs, y)

# Final comparison

sel=[p.fit_transform(Xs,y) for p in selector.values()]
features = [p.get_support(indices=True) for p in selector.values()]
features.append(elasticnet.get_support(indices=True))
total_feat=[]
for i in features:
	total_feat.append(np.array([F[k] for k in i]))


selector['ElasticNetCV']=[]
metrics = {k:f for k,f in zip(selector.keys(),total_feat)}
metrics = pd.DataFrame().from_dict(metrics, orient="index").T
metrics.to_csv("selection.csv")

# Compare FSS trough dendrograms/heatmap (graphic)
# Selected features shared between methods are used to better explain variability

# Rearranged and selected as 275-top. 
feat = pd.read_csv("topfeatures.csv")

## Create and export the working dataframe: scaled and reduced
cancer_sr = Xs[feat["Coincidence"]]
cancer_sr['Class'] = y
cancer_sr.to_csv("./Mix_BC_sr.csv")
