#!/usr/bin/env Rscript
library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(grid)
library(cowplot)
library(ggpubr)
library(pilot)
library(igraph)
library(ggraph)
library(ggdendro)
library(dendextend)

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

tileplot <- df %>%
	count(Method, Coincidence) %>%
	mutate(
#		Method = factor(Method, levels=a$Method),
		Coincidence = factor(Coincidence, levels=partition_leaves(dendro)[[1]])) %>%
	ggplot(aes(Method, Coincidence, fill= n)) + 
  geom_tile()+
  theme(
  	axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1),
  	axis.text.y=element_text(size=5),
  	legend.position="none")

# Plot a dendrogram
scores <- df %>%
	count(Method, Coincidence) %>%
	mutate(
#		Method = factor(Method, levels=a$Method),
		Coincidence = factor(Coincidence, levels=b$Coincidence)) %>%
		with(table(Coincidence, Method))

d = dist(scores, method = "binary")
hc = hclust(d, method="ward.D")
dendro <- as.dendrogram(hc)

dendroplot <- 
    ggdendrogram(
        dendro,
        rotate=TRUE)+
    theme(
    	axis.text.x = element_blank(),
    	axis.text.y = element_blank())

fss_comparison <- plot_grid(
	tileplot, 
	dendroplot, 
	align = "h",
	scale=c(1,1.096)
)
save_plot(
	"fss_comparison.pdf",
	fss_comparison,
	dpi=300,
	base_width = 2000,
	base_height = 14000,
	units = "px",
	useDingbats=FALSE
)

# Optional plots
# Circle Packing
library(packcircles)
library(ggplot2)

# For methods
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

# For coincidences
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
