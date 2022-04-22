#!/usr/bin/env Rscript
library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(ggthemes)
library(ggpubr)
# Descriptive analysis
## Load data
mix <- read_csv("../1_warehousing/Mix_BC.csv.gz")[,-c(1,2,8743)] %>%
## Agrupar en tres columnas
	pivot_longer(
		!V2,
		names_to = "aminoacidseq",
		values_to = "frequence") %>%
## transform to factor
	filter(frequence >= 0 & frequence< 1) %>%
	mutate(across(!frequence, as.factor)) %>%

mixbal <- read_csv("./Mix_BC_srbal.csv.gz") %>%
	pivot_longer(
		!Class,
		names_to = "aminoacidseq",
		values_to = "frequence") %>%
## transform to factor
	filter(frequence >= 0 & frequence< 1) %>%
	mutate(across(!frequence, as.factor))
	
levels(mix$V2) <- c("Negativo","Positivo")
levels(mixbal$Class) <- c("Negativo","Positivo")

# Imbalanced data barplot (before)
## Sampling proportion
## set.seed(123)
a <- mix %>% 
	group_by(aminoacidseq) %>% 
	slice_sample(prop = 0.1) %>%
	ggplot(aes(x=V2, y=frequence, fill = aminoacidseq))+
	geom_bar(position="stack", stat="identity")+
	xlab("¿Presenta cáncer?")+
	ylab("Frecuencia del oligopéptido")+
  theme(
  	legend.position="none",
    # Remove axis ticks and text
    axis.title = element_blank(),
    axis.ticks = element_blank(),
    # Use gray text for the region names
    axis.text.x = element_text(color = "gray12", size = 12),
    axis.text.y = element_text(color = "gray12", size = 12),
    # Move the legend to the bottom
    legend.position = "bottom",
    # Set default color and font family for the text
    text = element_text(color = "gray12"),
    # Make the background white and remove extra grid lines
    panel.background = element_rect(fill = "white", color = "white"),
    panel.grid = element_blank(),
    panel.grid.major.x = element_blank()
  )+
  # Make the guide for the fill discrete
  guides(
    fill = guide_colorsteps(
      barwidth = 50, barheight = .75, title.position = "top", title.hjust = .5
    )
  )+
  # New fill and legend title for number of tracks per region
  scale_fill_gradientn(
    "Accuracy",
     colours = c("#5E1E5B","#A62231","#F26F7E","#E95420","#F89B0F"),
     super = metR::ScaleDiscretised)

# Balanced data barplot (after)
## Sampling proportion
## set.seed(123)
b <- mixbal %>%	
	group_by(aminoacidseq) %>%
	slice_sample(prop = 0.1) %>%
	ggplot(aes(x=Class, y=frequence, fill = aminoacidseq))+
	geom_bar(position="stack", stat="identity")+
	xlab("¿Presenta cáncer?")+
	ylab("Frecuencia del oligopéptido")

# Merge two plots
fig <- ggarrange(a, 
					b,
					labels = c("A", "B"),
					ncol = 2, 
					nrow = 1)
ggsave("aabarplot.pdf", 
	width = 3200, 
	height = 1600,
	units = "px",
	useDingbats=FALSE)
