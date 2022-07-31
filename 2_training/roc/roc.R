#!/usr/bin/env Rscript
library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(plotROC)
# Generates a ROC curve with ggplot
pred <- read_csv("../predictions.csv") %>%
	select(svmrbf, lr, mlp, Reality) %>%
	pivot_longer(cols=!Reality, names_to="Modelo", values_to="Predicciones") %>%
	ggplot(aes(m = Predicciones, d = Reality, colour=Modelo))+
		geom_roc(n.cuts=20,labels=FALSE)+
		style_roc(theme = theme_grey)+
#		facet_wrap(~Modelo, scales="free")+
  # New fill and legend title for number of tracks per region
  scale_colour_manual(
    "Modelo",
     values = c("#5E1E5B","#F89B0F","#F26F7E"))+
  theme(
    # Use gray text for the region names
    axis.text.x = element_text(color = "gray12", size = 12),
    axis.text.y = element_text(color = "gray12", size = 12),
    # Move the legend to the bottom
    legend.position = "bottom",
    # Set default color and font family for the text
    text = element_text(color = "gray12")
    # Make the background white and remove extra grid lines
#    panel.background = element_rect(fill = "white", color = "grey")
#    panel.grid = element_blank(),
#    panel.grid.major.x = element_blank()
  )
ggsave("ROCs.pdf",
				pred, 
				dpi=300,
				width = 2000, 
				height = 2000, 
				units = "px",
				useDingbats=FALSE)
