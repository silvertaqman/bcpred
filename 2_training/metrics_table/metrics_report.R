#!/usr/bin/env Rscript
library(readr)
library(dplyr)
library(tidyr)
library(knitr)
library(ggplot2)
library(gt)
library(stringr)
library(pilot)

# Generates a table for summarise statistics
# read_csv("../validation_curve.csv") %>%
media <- read_csv("../validation_metrics.csv.gz") %>%
	select(-...1, -fit_time, -score_time) %>%
	group_by(model,method) %>%
	summarise(across(!folds, mean)) %>%
  gt(
  	rowname_col = "model",
  	groupname_col = "method"
  )

gtsave(media, "Medias.rtf")

sds <- read_csv("../validation_metrics.csv.gz") %>%
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
medias <- read_csv("../validation_metrics.csv.gz") %>%
	select(-...1, -fit_time, -score_time) %>%
	group_by(model,method) %>%
	summarise(across(!folds, mean))%>%
	pivot_longer(starts_with("test_"), names_to="Metric", values_to="media") %>%
	mutate(Metric = str_remove(Metric, "test_")) %>%
	mutate(model = factor(model))

sds <- read_csv("../validation_metrics.csv.gz") %>%
	select(-...1, -fit_time, -score_time) %>%
	group_by(model,method) %>%
	summarise(across(!folds, sd))%>%
	pivot_longer(starts_with("test_"), names_to="Metric", values_to="sd") %>%
	mutate(Metric = str_remove(Metric, "test_")) %>%
	mutate(model = factor(model))

# Heatmap for metrics
lims <- medias %>%
	inner_join(sds) %>%
	filter(method == "kfold") %>%
	group_by(model, Metric) %>%
	select(!method) %>%
	mutate(media = mean(media), sd=mean(sd))

heatmap <- lims %>%
	ggplot(aes(x=Metric, y=model, fill=media))+
		geom_tile()+
		geom_label(
			aes(label=paste0(100*round(media,3),"±",100*round(sd, 3))), colour="white")+
		scale_y_discrete(limits=unique(lims$model))+
		scale_x_discrete(position="top")+
		theme_pilot()
ggsave("tabla.png",heatmap,dpi=300, width = 2000, height = 2000,bg = "white", units = "px")	
