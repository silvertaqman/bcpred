#!/usr/bin/env Rscript
library(readr)
library(pilot)
library(tidyverse)
#####################################################################
theme_set(
	theme_pilot()+
	theme(
		legend.position = "bottom",
		legend.title=element_blank(),
		axis.text.x = element_text(
			color = "gray12", size = 12),
		axis.text.y = element_text(
			color = "gray12", size = 12),
		text = element_text(
			color = "gray12"),
		panel.background = element_rect(
			fill = "white",
			color = "white"),
		panel.grid = element_blank(),
		panel.grid.major.x = element_blank(),
		strip.background = element_blank()
		)
)
#####################################################################
# Descriptive analysis
## Load data
components <- read_csv("./PCAComponents.csv.gz") %>%
	select(!...1) %>% 
	rename_with(~ paste0("PC",1:351), everything())

variance <- read_csv("./PCAVarianceRatios.csv.gz") %>% 
	mutate(pc = 1:351, varatio = cumsum(`0`)) %>%
	select(pc, varatio)

# Cumulative Variance

ev <- variance %>% 
#	filter(varatio > 0.94, pc > 290) %>% 
	ggplot(aes(x=pc, y=varatio, group=1))+
		geom_line(color="blue")+
		geom_vline(xintercept = 275, color="blue")+
		geom_hline(yintercept = 0.93, color="red")+
		geom_bar(stat="identity",alpha=0.25,fill="red")+
		labs(x="Componente principal", y="Varianza Acumulada")+
		scale_x_continuous(breaks = sort(c(0,50,100,200,275,300)))+
		scale_y_continuous(breaks = sort(c(0,0.25,0.50,0.75,0.93,1.0)))

# Merge two plots
ggsave("varatio.png", 
	ev,
	width = 2500, 
	height = 1500,
	units = "px")
