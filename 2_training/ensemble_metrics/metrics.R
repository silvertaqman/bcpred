#!/usr/bin/env Rscript
library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(pilot)
library(ggpubr)
#############
# Data merging
#############
datos <- read_csv("bagging_validation_metrics.csv") %>%
	bind_rows(
		read_csv("boosting_validation_metrics.csv"),
		read_csv("voting_validation_metrics.csv"),
		read_csv("stacking_validation_metrics.csv")) %>%
	bind_cols(
		ensemble=rep(
			c("bagging","boosting","voting","stacking"),
			each=180))

#############
# Set general theme
#############
theme_set(
	theme(
		  # Use gray text for the region names
		  axis.text.x = element_text(angle=90,color = "gray12", size = 12),
		  axis.text.y = element_text(color = "gray12", size = 12),
		  # Set default color and font family for the text
		  text = element_text(color = "gray12"),
		  # Make the background white and remove extra grid lines
		  panel.background = element_rect(fill = "white", color = "white"),
		  panel.grid = element_blank(),
		  panel.grid.major.x = element_blank(),
		  strip.background = element_blank()
#		  strip.text.x = element_blank()
		  )+
  	theme_pilot()
)

##################################################################################
# barplot for model selection
##################################################################################
barmetrix <- function(x){
	bam <- x %>%
		select(!ends_with("_time"), !...1) %>%
		pivot_longer(
			cols=starts_with("test_"), 
			names_to="Test", 
			values_to="Metric") %>%
		mutate_if(is.character, as.factor) %>%
		group_by(Test, model) %>%
		summarize(across(Metric, mean))
		bam <- ggplot(bam, aes(x=model, y=Metric/4, fill=Test))+
				geom_bar(stat="identity")+
		coord_flip()+
		ylab("Metrics")+
  scale_fill_pilot()+
		scale_x_discrete(limits=c("STACK3","STACK2","STACK1","HTE","SOFT","MLP","HARD","LR","SVM","DTC"))
		return(bam)
}
barmetrix(datos)

ggsave(
	"BestModel.pdf",
	barmetrix(datos),
	dpi=300,
	width = 4000, 
	height = 2000, 
	units = "px",
	useDingbats=FALSE)

##################################################################################
# Boxplot of metrics comparison
##################################################################################
boxmetrix <- function(x){
	bxm <- x %>%
		select(!ends_with("_time")) %>%
		pivot_longer(
			cols=starts_with("test_"), 
			names_to="Test", 
			values_to="Metric") %>%
		mutate_if(is.character, as.factor) %>%
		ggplot(aes(x=model, y=Metric, fill=factor(method)))+
				geom_boxplot()+
#				geom_hline(yintercept=0.94)+
		facet_grid(ensemble~Test)+
		scale_fill_manual(
		  "Modelo",
		   values = c("#F89B0F","#F26F7E"))
		return(bxm)
}
boxmetrix(datos)

# Save all figures in pdf

ggsave(
	"Metrics.pdf",
	boxmetrix(datos),
	dpi=300,
	width = 4000, 
	height = 2000, 
	units = "px",
	useDingbats=FALSE)

#############################33
# ROC-Curve
###############################
library(plotROC)
library(pilot)
# Generates a ROC curve with ggplot
pred <- read_csv("../predictions.csv") %>%
	select(!...1) %>%
	pivot_longer(cols=!Reality, names_to="Modelo", values_to="Predicciones") %>%
	ggplot(aes(m = Predicciones, d = Reality, colour=Modelo))+
		geom_roc(n.cuts=20,labels=FALSE)+
		style_roc(theme = theme_grey)+
		scale_color_pilot()+
		geom_rocci(linetype = 1)

# Los metodos STACK1, AdaDTC y AdaSVM tiene los AUC mas altos
positions<-arrange(calc_auc(pred),desc(AUC))
options(digits=4)
pred <- read_csv("../predictions.csv") %>%
	select(!...1) %>%
	pivot_longer(cols=!Reality, names_to="Modelo", values_to="Predicciones") %>%
	ggplot(aes(m = Predicciones, d = Reality, colour=Modelo))+
		geom_roc(n.cuts=20,labels=FALSE)+
		style_roc(theme = theme_grey)+
		scale_color_pilot(
			breaks=positions$Modelo, 
			labels = paste(positions$Modelo, positions$AUC))

ggsave("ROCs.pdf",
				pred, 
				dpi=300,
				width = 2000, 
				height = 2000, 
				units = "px",
				useDingbats=FALSE)

