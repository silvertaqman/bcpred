#!/usr/bin/env Rscript
library(tidyverse)
library(pilot)
library(readr)
library(knitr)
library(cowplot)
library(patchwork)
library(ggpubr)
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
datos <- read_csv("./minivalidation_metrics.csv")
barmetrix <- function(x){
	bam <- x %>%
		select(!ends_with("_time"), !...1) %>%
		pivot_longer(
			cols=starts_with("test_"), 
			names_to="Test", 
			values_to="Metric") %>%
		mutate_if(is.character, as.factor) %>%
		group_by(Test, model) %>%
		summarize(across(Metric, mean)) %>%
		arrange(desc(Metric))
		
	order <- bam %>%
		group_by(model) %>%
		summarize(across(Metric,sum)) %>%
		arrange(Metric)
		
		bam <- ggplot(bam, aes(x=model, y=Metric, fill=Test))+
			geom_bar(stat="identity")+
			coord_flip()+
			ylab("Metrics")+
  	scale_fill_pilot()+
  	geom_hline(aes(yintercept=0),colour="red",linewidth=1.5)+
  	scale_x_discrete(limits = order$model)
	return(bam)
}
# Hard_ensemble, adarbf and bagmlp are top3

#ggsave("Metrics.png",barmetrix(datos),dpi=320,width = 2000, height = 1500,bg = "white", units = "px")
##################################################################################
# Boxplot of metrics comparison
##################################################################################
boxmetrix <- function(x){
	bxm <- x %>%
		select(!(ends_with("_time")|...1)) %>%
		pivot_longer(
			cols=starts_with("test_"), 
			names_to="Test", 
			values_to="Metric") %>%
		mutate_if(is.character, as.factor) %>%
		ggplot(aes(x=model, y=Metric, fill=factor(method)))+
				geom_boxplot()+
#				geom_hline(yintercept=0.94)+
		facet_wrap(~Test, scales="free_y")+
		scale_fill_pilot()+labs(fill = 'Method')
		return(bxm)
}

#ggsave("Metrics2.png",boxmetrix(datos),dpi=320,width = 2000, height = 1500,bg = "white", units = "px")

# all merged
all <- (barmetrix(datos)/boxmetrix(datos))+
	plot_layout(guides = 'collect')+
	plot_annotation(tag_levels="A")
ggsave("all.png",all,dpi=320, width = 4000, height = 4000,bg = "white", units = "px")
