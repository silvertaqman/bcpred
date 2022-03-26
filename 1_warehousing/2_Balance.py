#!/usr/bin/env python3
import numpy as np
import pandas as pd
import sklearn
import imblearn

# Load data
cancer = pd.read_csv("Mix_BreastCancer_sr.csv")

#Select data with an X to y model
y = cancer['Class']
X = cancer.drop('Class', axis = 1)
Features = list(X.columns)

# SMOTE
from imblearn.over_sampling import SMOTE
smote = SMOTE(ratio='minority',random_state=123)
Xs, ys = smote.fit_resample(X.values, y)
    
# Export smoted, balanced data
cancer = pd.DataFrame(Xs,columns=Features)
cancer['Class'] = ys # add class column
cancer.to_csv('Mix_BreastCancer_srbal', index=False)
