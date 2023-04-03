# Spencer

This repository contains the workflow adn data required to train a Multilyer Percetron based on Biomarker Identification on Protein Sequences obtained through MALDI-TOF Mass Spectrometry. Also, contains the ready-to-use ML model accesible through Python (>=3.9) and related packages. 

## Usage

AccNo. CIDH920105  Normalized average hydrophobicity scales (Cid et al., 1992)
AccNo. BHAR880101  Average flexibility indices (Bhaskaran-Ponnuswamy, 1988)
AccNo. CHAM820101  Polarizability parameter (Charton-Charton, 1982)
AccNo. CHAM820102  Free energy of solution in water, kcal/mole (Charton-Charton, 1982)
AccNo. CHOC760101  Residue accessible surface area in tripeptide (Chothia, 1976)
AccNo. BIGC670101  Residue volume (Bigelow, 1967)
AccNo. CHAM810101  Steric parameter (Charton, 1981)
AccNo. DAYM780201  Relative mutability (Dayhoff et al., 1978b)

## Describe relation between proteic composition and breast cancer 

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

