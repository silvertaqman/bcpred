#!/usr/bin/env python3
# Preload packages
import sys
import pandas as pd
import numpy as np
import matplotlib as plt
import scipy
import IPython
import sklearn
import mglearn
import joblib

# Data loading
bc = pd.read_csv("./Mix_BC_srbal.csv.gz")
bc_input = bc.iloc[0:466, 0:300]
bc_output = bc['Class']

# Load models
svmlin = joblib.load("./models/bc_svmlin.pkl")
svmrbf = joblib.load("./models/bc_svmrbf.pkl")
lr = joblib.load("./models/bc_lr.pkl")
mlp = joblib.load("./models/bc_mlp.pkl")
