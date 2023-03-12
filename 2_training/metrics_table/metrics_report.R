#!/usr/bin/env Rscript
library(readr)
library(dplyr)
library(tidyr)
library(knitr)
library(ggplot2)
library(gt)
library(stringr)

# Generates a table for summarise statistics
# read_csv("../validation_curve.csv") %>%
media <- read_csv("../validation_metrics.csv") %>%
	select(-...1, -fit_time, -score_time) %>%
	group_by(model,method) %>%
	summarise(across(!folds, mean)) %>%
  gt(
  	rowname_col = "model",
  	groupname_col = "method"
  )

gtsave(media, "Medias.rtf")

sds <- read_csv("../validation_metrics.csv") %>%
	select(-...1, -fit_time, -score_time) %>%
	group_by(model,method) %>%
	summarise(across(!folds, sd)) %>%
  gt(
  	rowname_col = "model",
  	groupname_col = "method"
  )

gtsave(sds, "Desviaciones.rtf")

#gtsave("tab_1.tex"), gtsave("tab_1.rtf")

# grafico de intervalos de confianza (best ensembles are stack_1 and bagmlp)
medias <- read_csv("../validation_metrics.csv") %>%
	select(-...1, -fit_time, -score_time) %>%
	group_by(model,method) %>%
	summarise(across(!folds, mean))%>%
	pivot_longer(starts_with("validation_"), names_to="Metric", values_to="media") %>%
	mutate(Metric = str_remove(Metric, "validation_"))

sds <- read_csv("../validation_metrics.csv") %>%
	select(-...1, -fit_time, -score_time) %>%
	group_by(model,method) %>%
	summarise(across(!folds, sd))%>%
	pivot_longer(starts_with("validation_"), names_to="Metric", values_to="sd") %>%
	mutate(Metric = str_remove(Metric, "validation_"))

lims <- medias %>%
	inner_join(sds) %>%
	mutate(maximo = media + sd, minimo = media - sd) %>%
	filter(model %in% c("stack_1", "bagmlp"))%>% #, media > 0.975
	arrange(minimo, decrease=TRUE)

p <- ggplot(lims, aes(model, media, fill = Metric))+
	facet_grid(Metric~method, scales="free_y")
p + geom_errorbar(aes(ymin = minimo, ymax = maximo), width = 0.2)
