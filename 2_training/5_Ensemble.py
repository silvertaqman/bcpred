#!/usr/bin/env python3
###################################################################
# Preload packages
###################################################################
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import IPython
import sklearn
import mglearn
import joblib
import itertools
import vecstack

###################################################################
# Data loading
###################################################################
bc = pd.read_csv("./Mix_BC_srbal.csv.gz")
bc_input = bc.iloc[0:466, 0:300]
bc_output = bc['Class']

# Metrics (Every model)
from sklearn.metrics import confusion_matrix,accuracy_score,roc_auc_score,recall_score, mean_squared_error,log_loss

# Data partition (mathematical notation)
from sklearn.model_selection import train_test_split as tts
X, Xt, y, yt = tts(bc_input,bc_output,random_state=74)

# Loading models
#from sklearn.svm import SVC
#from sklearn.neural_network import MLPClassifier
#from sklearn.linear_model import LogisticRegression
svmrbf = joblib.load("./models/bc_svmrbf.pkl")
lr = joblib.load("./models/bc_lr.pkl")
mlp = joblib.load("./models/bc_mlp.pkl")
models = [svmrbf, lr, mlp]

# Training and Tuning ensembles for final selection
###################################################################
# Bagging
###################################################################
# evaluate bagging ensemble for classification
from sklearn.model_selection import cross_validate as cv
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import BaggingClassifier

# define the model
bagrbf = BaggingClassifier(svmrbf, random_state=74).fit(X,y)
baglr = BaggingClassifier(svmrbf, random_state=74).fit(X,y)
bagmlp = BaggingClassifier(svmrbf, random_state=74).fit(X,y)
bagmodels = [bagrbf, baglr, bagmlp]

# K-fold Validation
kfcv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=74)
scoring = ['accuracy','recall','precision','roc_auc']
kfv = [cv(p, bc_input, bc_output, scoring=scoring, cv=kfcv, n_jobs=-1, error_score='raise') for p in bagmodels]

# K-stratified Validation
ksfcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=74)
ksfv = [cv(p, bc_input, bc_output, scoring=scoring, cv=ksfcv, n_jobs=-1, error_score='raise') for p in bagmodels]
metrics = list(itertools.chain.from_iterable(zip(kfv, ksfv)))

# Exporting metrics to csv
metrics = pd.concat(map(pd.DataFrame, (metrics[i] for i in range(len(metrics)))))
metrics['repeat'] = 30*['fold'+str(i+1) for i in range(3)]
metrics['folds'] = 18*['fold'+str(i+1) for i in range(5)]
model = np.repeat(['svmrbf', 'lr', 'mlp'], 5)
metrics['model'] = np.tile(model, 6)
metrics['method'] = np.repeat(['kfold','stratified'],45)
metrics.to_csv('./ensemble_metrics/bagging_validation_metrics.csv')

# Export data for overfit learning curve
from sklearn.model_selection import learning_curve
size_svm, score_svm, tscore_svm, ft_svm,_ = learning_curve(bagrbf, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10),return_times=True)
size_lr, score_lr, tscore_lr, ft_lr,_ = learning_curve(baglr, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10),return_times=True)
size_mlp, score_mlp, tscore_mlp, ft_mlp,_ = learning_curve(bagmlp, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10),return_times=True)

metrics = pd.DataFrame()
metrics['train_size'] = np.concatenate((size_svm, size_lr, size_mlp))
metrics['models'] = 10*["svm"]+10*['lr']+10*['mlp']
metrics = pd.concat([metrics,pd.DataFrame(np.concatenate([score_svm, score_lr, score_mlp])), pd.DataFrame(np.concatenate([tscore_svm, tscore_lr, tscore_mlp])),pd.DataFrame(np.concatenate([ft_svm, ft_lr, ft_mlp]))],axis=1)
metrics.columns = ['train_size','models','train_scores_fold1','train_scores_fold2','train_scores_fold3','train_scores_fold4','train_scores_fold5','test_scores_fold1','test_scores_fold2','test_scores_fold3','test_scores_fold4','test_scores_fold5','fit_times_fold1','fit_times_fold2','fit_times_fold3','fit_times_fold4','fit_times_fold5']
metrics.to_csv('./ensemble_metrics/bagging_learning_curve.csv')

###################################################################
# Mixing combinations
# Boosting: training over weak classifiers
###################################################################

# Adaboost
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_validate as cv
from sklearn.metrics import make_scorer, accuracy_score, recall_score, roc_auc_score, confusion_matrix, precision_score
adarbf = AdaBoostClassifier(base_estimator=SVC(kernel='rbf',probability=True, random_state=74))
params = {
	'n_estimators': (1,3,5,10,20,25,40,50,75,90,100),                  
	'learning_rate': (0.0001, 0.001, 0.01, 0.1, 1.0,1.5),
	'algorithm': ('SAMME', 'SAMME.R'),
	'base_estimator__C': (0.1,0.5,1.0,5.0,10.0),
	'base_estimator__gamma': (1.0,0.75,0.5,0.25,0.1,0.01)}
gs = GridSearchCV(adarbf, param_grid, cv=5).fit(X,y) # lot of time

svmrbf = SVC().fit(X,y)
adarbf = AdaBoostClassifier(**gs.best_params_).fit(X,y)

scoring = ['accuracy','recall','precision','roc_auc']
kfv = cv(adarbf, bc_input,bc_output,cv=5,scoring= scoring, n_jobs=-1)


clf = GridSearchCV(adarbf,params, cv=5, scoring='accuracy', n_jobs=-1)
    clf.fit(iris.data, iris.target)
    print("Accuracy boost_SVM = ",clf.best_score_)
    print("Best params boost_SVC = ", clf.best_params_)
    # Best params boost_SVC =  {'algorithm': 'SAMME.R', 'base_estimator__C': 0.5,
    # 'base_estimator__gamma': 0.75, 'learning_rate': 0.1, 'n_estimators': 20}

    
    boost_SVC.fit(iris.data, iris.target)
    print("Accuracy SVC= " ,boost_SVC.score(iris.data, iris.target))
    
    
def train_gridsearch_classification_LogReg(iris,cv_kf):
    
    boost_LogReg = AdaBoostClassifier(base_estimator=LogisticRegression(max_iter = 1000))
    
    parameters = {'n_estimators': (1,2,3,4,5,6,7,8,9,10,15,20,25),                  
                  'learning_rate': (0.0001, 0.001, 0.01, 0.1, 1.0),
                  'algorithm': ('SAMME', 'SAMME.R'),
                  'base_estimator__C':(0.1,0.5,1.0,5.0, 10.0),
                  'base_estimator__solver': ('newton-cg','lbfgs')}
    
    
    clf = GridSearchCV(boost_LogReg,parameters, cv=cv_kf ,scoring='accuracy', n_jobs=-1)    
    clf.fit(iris.data, iris.target)
    print("Accuracy boost_SVM = ",clf.best_score_)
    print("Best params boost_LogReg = ", clf.best_params_)
    # Best params boost_LogReg =  {'algorithm': 'SAMME', 'base_estimator__C': 1.0, 
    # 'base_estimator__solver': 'newton-cg', 'learning_rate': 0.1, 'n_estimators': 25}
    
    boost_LogReg.fit(iris.data, iris.target)
    print("Accuracy Log_Reg= " ,boost_LogReg.score(iris.data, iris.target))
    
#==============================================================================

cv_kf = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
iris_dt   = load_iris()

#train_gridsearch_classification_DTC(iris_dt,cv_kf)
#train_gridsearch_classification_SVC(iris_dt,cv_kf)
train_gridsearch_classification_LogReg(iris_dt,cv_kf)

# Ada + DTC = 0.964
# Ada + SVC = 0.9733
# Ada + LR  = 0.964


###################################################################
# Stacking: train multiple models together
###################################################################
from vecstack import stacking
# computing the stack features
s, st = stacking(models, X, y, Xt, regression = True, n_folds = 4, shuffle = True, random_state = 74)
# fitting the second level model with stack features
allplus = svmrbf.fit(s, y)
# predicting the final output using stacking
yp = allplus.predict(st)
# printing the root mean squared error between real value and predicted value
print("MSE: ", mean_squared_error(yt, yp))

# Max/Hard Voting
from sklearn.ensemble import VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
estimators = [('radial',CalibratedClassifierCV(svmrbf)),('logistic',lr),('multi',mlp)]
hard_ensemble = VotingClassifier(estimators, voting='hard').fit(X,y)
hard_ensemble.score(Xt,yt)

# printing log loss between actual and predicted value

print("log_loss: ", log_loss(yt, yp))

# Average/Soft Voting

yp = sum([p.predict(Xt) for p in models])/3.0

soft_ensemble = VotingClassifier(estimators, voting='soft').fit(X,y)
yp = soft_ensemble.predict(Xt)

# printing the root mean squared error between real value and predicted value
print("MSE of model 1: ", mean_squared_error(yt, svmrbf))
print("MSE of model 2: ", mean_squared_error(yt, lr))
print("MSE of model 3: ", mean_squared_error(yt, mlp))

# printing the root mean squared error between real value and predicted value
print("Ensemble MSE", mean_squared_error(yt, yp))

t = np.arange(0,len(svmrbf))
plt.plot(t, yt, c='blue')
#plt.plot(t, pred_1,c='red')
#plt.plot(t, pred_2,c='green')
#plt.plot(t, pred_3,c='DarkBlue')
plt.plot(t, yp, c='yellow')
plt.legend(["Test", "Prediction"], loc ="lower right")
plt.grid()
plt.show()
