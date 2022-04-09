#!/usr/bin/env Rscript
library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(ggthemes)
library(ggpubr)
# Descriptive analysis
## Load data
mix <- read_csv("../1_warehousing/Mix_BreastCancer.csv")[,-c(1,2,8743)] %>%
## Agrupar en tres columnas
	pivot_longer(
		!V2,
		names_to = "aminoacidseq",
		values_to = "frequence") %>%
## transform to factor
	filter(frequence >= 0 & frequence< 1) %>%
	mutate(across(!frequence, as.factor)) %>%

mixbal <- read_csv("./Mix_BreastCancer_srbal.csv") %>%
	pivot_longer(
		!Class,
		names_to = "aminoacidseq",
		values_to = "frequence") %>%
## transform to factor
	filter(frequence >= 0 & frequence< 1) %>%
	mutate(across(!frequence, as.factor))
	
levels(mix$V2) <- c("Sano","Enfermo")
levels(mixbal$Class) <- c("Sano","Enfermo")

# Imbalanced data barplot (before)
## Sampling proportion
## set.seed(123)
a <- mix %>% 
	group_by(aminoacidseq) %>% 
	slice_sample(prop = 0.1) %>%
	ggplot(aes(x=V2, y=frequence, fill = aminoacidseq))+
	geom_bar(position="stack", stat="identity")+
	theme_tufte(base_size = 12, base_family = "serif")+
	theme(legend.position="none")+
	xlab("¿Presenta cáncer?")+
	ylab("Frecuencia del oligopéptido")
# Balanced data barplot (after)
## Sampling proportion
## set.seed(123)
b <- mixbal %>%	
	group_by(aminoacidseq) %>%
	slice_sample(prop = 0.1) %>%
	ggplot(aes(x=Class, y=frequence, fill = aminoacidseq))+
	geom_bar(position="stack", stat="identity")+
	theme_tufte(base_size = 12, base_family = "serif")+
	theme(legend.position="none")+
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
