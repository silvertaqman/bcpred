#!/usr/bin/env Rscript
library(readr)
library(dplyr)
library(tidyr)
library(knitr)

# Generates a table for summarise statistics
# read_csv("../validation_curve.csv") %>%
tabla <- read_csv("validation_metrics.csv") %>%
	select(-...1, -fit_time, -score_time) %>%
	group_by(model,method) %>%
	summarise(across(!folds, mean)) %>%
  gt(
  	rowname_col = "model",
  	groupname_col = "method"
  ) %>%
  cols_label(
       test_accuracy="Accuracy",
       test_recall="Recall",
       test_precision="Precision",
       test_roc_auc="AUROC")

gtsave(tabla, "estadisticos.rtf")
# gtsave("tab_1.tex"), gtsave("tab_1.rtf")
