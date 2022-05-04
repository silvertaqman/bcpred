#!/usr/bin/env Rscript
library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(ggpubr)
library(purrr)
# Descriptive analysis
## Load data
mix <- read_csv("./Mix_BC.csv.gz")[,-c(1,2,8743)] %>%
## Agrupar en tres columnas
	pivot_longer(
		!V2,
		names_to = "aminoacidseq",
		values_to = "frequence") %>%
## transform to factor
	filter(frequence > 0 & frequence< 1) %>%
	mutate(Class = V2) %>%
	mutate(
		group = rep('Original', 408073),
		property = pmap(
			., 
			~ifelse(
				nchar(..2) <= 3,
				'Composición',
				'Atributo')
				) %>% 
				unlist
			) %>% 
	select(group,property, frequence, Class) 

mixbal <- read_csv("../2_training/Mix_BC_srbal.csv.gz") %>%
	pivot_longer(
		!Class,
		names_to = "aminoacidseq",
		values_to = "frequence") %>%
## transform to factor
	filter(frequence > 0 & frequence< 1) %>%
	mutate(
		group = rep('Balanceado', 13298),
		property = pmap(
			.,
			~ifelse(
				nchar(..2) <= 3,
				'Composición',
				'Atributo')
				) %>%
				unlist
			) %>%
	select(group,property, frequence, Class)

# Merge datasets

mix <- mix %>%
	bind_rows(mixbal) %>%
	mutate(across(!frequence, factor))
rm(mixbal)

levels(mix$Class) <- c("Negativo","Positivo")

# Data barplot (before)

a <- mix %>%
	count(group, property, Class) %>%
	ggplot(aes(x=Class,
		y=n,
		fill = property))+
	geom_bar(position="stack", stat="identity")+
  # New fill and legend title for number of tracks per region
  scale_fill_manual(
    "Tipo de Variable",
     values = c("#5E1E5B","#F89B0F")
     )+
  xlab("¿Presenta cáncer?")+
	ylab("Frecuencia")+
  theme(
  	legend.position="bottom",
#    axis.title = element_blank(),
    axis.text.x = element_text(color = "gray12", size = 12),
    axis.text.y = element_text(color = "gray12", size = 12),
    text = element_text(color = "gray12"),
    panel.background = element_rect(fill = "white", color = "white"),
    panel.grid = element_blank(),
    panel.grid.major.x = element_blank()
  )+
  facet_wrap(~group, scales = 'free')+
  geom_label(
  	aes(label = n),
  	position = position_stack(vjust = 0.5),
  	colour = "white",
  	fontface = "bold")

# Merge two plots
ggsave("aabarplot.pdf", 
	width = 3200, 
	height = 1600,
	units = "px",
	useDingbats=FALSE)
