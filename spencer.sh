#!/bin/bash
# The reproducible workflow for bcpred as a script
cd ./1_warehousing
python3 1_CleanScale.py
pigz ./Mix_BC_sr.csv
python3 2_Balance.py
cd ../2_training
pigz ./Mix_BC_srbal.csv
python3 ./3_Train.py
cd ./gridsearch
pigz linear.csv logistic.csv mlp.csv rbf.csv
Rscript gridsearch.R


