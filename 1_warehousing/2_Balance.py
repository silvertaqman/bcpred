#!/usr/bin/env python3
import numpy as np
import pandas as pd
import sklearn
import imblearn

# Load data
cancer = pd.read_csv("./Mix_BC_sr.csv.gz")

#Select data with an X to y model
y = cancer['Class']
X = cancer.drop('Class', axis = 1)
Features = list(X.columns)

# SMOTE
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=74)
Xs, ys = smote.fit_resample(X.values, y)
    
# Export smoted, balanced data
cancer = pd.DataFrame(Xs,columns=Features)
cancer['Class'] = ys # add class column
cancer.to_csv('Mix_BreastCancer_srbal.csv', index=False)
smote = SMOTE(random_state=74)
X, y = smote.fit_resample(X.values, y)
    
# Export smoted, balanced data
cancer = pd.DataFrame(X,columns=Features)
cancer['Class'] = y # add class column
cancer.to_csv('../2_training/Mix_BC_srbal.csv', index=False)
