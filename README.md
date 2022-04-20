# BCPred

Predict breast cancer related proteins
This repo cointains all the efforts to generate an ML ensemble-based prediction system with high accuracy and recall. In order to do so, the files are ordered inside three directories: 

## warehousing
Here is the data warehousing process. It includes: outliers removal, minmax scaling and invariant columns removal. 

Data from [Soto, (2020)](https://github.com/muntisa/neural-networks-for-breast-cancer-proteins/tree/master/datasets) is recopiled and the process for data warehousing is replicated as a solo script. Relational data format is setted up. A minimax scaler is adapted for every column and then exported as *.pkl* file. Duplicated entries are removed, and then invariant columns, reducing features from 8742 to 8709. A PCA approach is applied and concludes more than 97% of the variance in class response is explained with 300 columns. Feature Subset Selection is applied, and unselected features are exported.   

## training

The four best classifiers from muntisa paper are selected, trained and exported: svm with **linear** and **radial** kernel, **logistic regression** and **multilayer perceptron**. Cross-validation and stratified cross-validation processes are executed to estimate accuracy per model. Then, accuracy, recall and roc-auc scores are calculated from the confusion matrix and are exported. 

The ensemble process ...

## evaluating

The refined model is evaluated with three datasets. 

## ensembling

#### About technical support: CEDIA
https://antiguo.cedia.edu.ec/es/servicios/tecnologia/infraestructura/cluster-para-hpc/como-trabajar-en-un-nodo-hpc-cedia

