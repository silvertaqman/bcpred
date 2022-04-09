#!/bin/bash
# The automated workflow for bcpred
python3 ./1_warehousing/1_CleanScale.py
pigz ./1_warehousing/Mix_BreastCancer_sr.csv
python3 ./1_warehousing/2_Balance.py
pigz ./2_training/Mix_BreastCancer_srbal.csv
python3 ./2_training/3_TrainValidate.py
