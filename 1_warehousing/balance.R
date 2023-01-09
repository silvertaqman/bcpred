#!/usr/bin/env Rscript
library(readr)
library(ggpubr)
library(purrr)
library(pilot)
library(tidyverse)
library(e1071)
library(pilot)
library(knitr)
library(grid)
library(cowplot)
library(igraph)
library(ggraph)
library(ggdendro)
library(dendextend)
library(patchwork)
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
				nchar(..2) <= 3, # only frequences have this kind of names
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

levels(mix$Class) <- c("Control","Paciente")
mix$group <- relevel(mix$group, ref="Original")

# Data barplot (before)

warehouse <- mix %>%
	count(group, property, Class) %>%
	ggplot(aes(x=Class,
		y=n,
		fill = property))+
	geom_bar(position="stack", stat="identity")+
  facet_wrap(~group, scales = 'free')+
  geom_label(
  	aes(label = n),
  	position = position_stack(vjust = 0.5),
  	colour = "white",
  	fontface = "bold")+
  	labs(x="Estado",y="Frecuencia",legend="Tipo de Variable")+
  	scale_fill_pilot()

# Merge two plots
ggsave("aabarplot.pdf", 
	warehouse,
	width = 3200, 
	height = 1600,
	units = "px",
	useDingbats=FALSE)

# FSS
# TilePlot
df <- read_csv("./selection.csv.gz") %>%
        select(!...1) %>%
        pivot_longer(
                everything(),
                names_to = "Method",
                values_to = "Coincidence") %>%
        drop_na(Coincidence) %>%
        mutate_if(is.character, as.factor)

# order of files
a <- df %>%
        count(Method) %>% 
        arrange(desc(n))

b <- df %>%
        count(Coincidence) %>% 
        arrange(desc(n))

# Plot a dendrogram
scores <- df %>%
	count(Method, Coincidence) %>%
	mutate(
#		Method = factor(Method, levels=a$Method),
		Coincidence = factor(Coincidence, levels=b$Coincidence)) %>%
		with(table(Coincidence, Method))

scores <- scores[rowSums(scores)>5,]

# export scores with > 5 frequences
read_csv("Mix_BC.csv.gz") %>%
	select(any_of(rownames(scores))) %>%
	write_csv("Mix_BC_selected.csv")

# dendrogram

d = dist(scores, method = "binary")
hc = hclust(d, method="ward.D")
dendro <- as.dendrogram(hc)

tileplot <- df %>%
	count(Method, Coincidence) %>%
	mutate(
#		Method = factor(Method, levels=a$Method),
		Coincidence = factor(Coincidence, levels=partition_leaves(dendro)[[1]])) %>%
	drop_na() %>%
	ggplot(aes(Method, Coincidence, fill= n)) + 
  geom_tile()+
  theme(
  	axis.text.x=element_text(angle=90,size=5),
  	axis.text.y=element_text(size=5),
  	legend.position="none")+
  labs(x="Métodos", y="Coincidencias")

dendroplot <- 
    ggdendrogram(
        dendro,
        rotate=TRUE)+
    theme_classic()+
    theme(
    	axis.text.y = element_blank(),
    	axis.ticks = element_blank(),
    	axis.line = element_blank()
    	)+
    labs(y="Distancia",x="")

fss_comparison <- plot_grid(
	tileplot, 
	dendroplot, 
	rel_widths=c(3,1),
	rel_heights=c(2,2),
	align = "h",
	scale=c(1,1.066)
)
layout<-c(area(1,1,2,1), area(1,2,3,3))
balance <- warehouse+fss_comparison+
	plot_layout(design=layout)+
	plot_annotation(tag_levels="A")
	
ggsave("balance.pdf",
				balance, 
#				device = "pdf",
				dpi="retina",
				width = 4000, 
				height = 2200,
#				bg = "white", 
				units = "px",
				useDingbats=FALSE)
#ggsave("balance.png",balance,dpi=320,width = 4000, height = 2200,bg = "white", units = "px")

# Optional plots

# Circle Packing
library(packcircles)
# By methods
packing <- circleProgressiveLayout(a$n, sizetype='area')
data <- cbind(a, packing)
dat.gg <- circleLayoutVertices(packing, npoints=50)

ggplot() +   
  geom_polygon(
  	data = dat.gg, 
  	aes(x, y, group = id, fill=as.factor(id)), 
  	colour = "black", 
  	alpha = 0.6) +
  geom_text(
  	data = data, 
  	aes(x, y, size=n, label = Method)) +
  scale_size_continuous(range = c(1,4)) +
  theme_pilot() + 
  theme(legend.position="none") +
  coord_equal()

# By coincidences
packing <- circleProgressiveLayout(b$n, sizetype='area')
data <- cbind(b, packing)
dat.gg <- circleLayoutVertices(packing, npoints=50)

ggplot() +   
  geom_polygon(
  	data = dat.gg, 
  	aes(x, y, group = id, fill=as.factor(id)), 
  	colour = "black", 
  	alpha = 0.6) +
  geom_text(
  	data = data, 
  	aes(x, y, size=n, label = Coincidence)) +
  scale_size_continuous(range = c(1,4)) +
  theme_pilot() + 
  theme(legend.position="none") +
  coord_equal()
