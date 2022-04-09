#!/usr/bin/env python3
import numpy as np
import pandas as pd
import sklearn

#Load data
cancer = pd.read_csv("Mix_BreastCancer.csv.gz")
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

## Remove duplicated values
cancer_s.drop_duplicates(keep=False, inplace=True)

## Remove invariant columns
y = cancer_s['Class']
X = cancer_s.drop('Class', axis = 1)
Features = list(X.columns)

### Selected features
from sklearn.feature_selection import VarianceThreshold
selector= VarianceThreshold()
Xdata = selector.fit_transform(X.values)
sf = []
[sf.append(Features[i]) for i in selector.get_support(indices=True)]

## Create and export the working dataframe: scaled and reduced
cancer_sr = pd.DataFrame(Xdata,columns=sf)
cancer_sr['Class'] = y
cancer_sr.to_csv("./Mix_BreastCancer_sr.csv.gz", index=False)

'''
Not applied yet: Remove dependent columns by PCA: 8709 to 322
y = cancer_s_reduced_selected['Class']
X = cancer_s_reduced_selected.drop('Class', axis = 1)
Features = list(X.columns)
from sklearn.decomposition import PCA
pca = PCA(.96)
pca.fit(cancer_s_reduced_selected)
reduced = pca.transform(cancer_s_reduced_selected)
cancer_s_reduced_selected_independent = pd.DataFrame(reduced, columns =cancer_s.columns)
No se como obtener los nombres de dichas columnas. 
'''
