#!/usr/bin/env Rscript
library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(grid)
library(ggpubr)
library(pilot)
library(igraph)
library(ggraph)
library(ggdendro)

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

tile <- df %>%
	count(Method, Coincidence) %>%
	mutate(
		Method = factor(Method, levels=a$Method),
		Coincidence = factor(Coincidence, levels=b$Coincidence)) %>%
	ggplot(aes(Method, Coincidence, fill= n)) + 
  geom_tile()+
  theme(
  	axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1),
  	axis.text.y=element_text(size=5),
  	legend.position="none")

ggsave("tile.pdf", 
	tile,
	width=1500,
	height= 12000, 
	units="px", 
	useDingbats=FALSE)

# Plot a dendrogram

scores <- df %>%
	count(Method, Coincidence) %>%
	mutate(
		Method = factor(Method, levels=a$Method),
		Coincidence = factor(Coincidence, levels=b$Coincidence)) %>%
		with(table(Coincidence, Method))

d = dist(scores, method = "binary")
hc = hclust(d, method="ward.D")
pdf("dendrogram.pdf", width=45, height=10)
plot(hc, cex=0.5)
dev.off()

# Merge them requires to change order
long <- df %>% 
	count(Method, Coincidence) %>%
	mutate(
		Coincidence = factor(
			Coincidence,
			levels=partition_leaves(dendro)[[1]]))
dendro <- as.dendrogram(hc)

longtile <- long %>%
	ggplot(aes(Method, Coincidence, fill= n)) + 
  geom_tile()+
  theme(
  	axis.text.x = element_text(
  		angle = 90,
  		vjust = 0.5,
  		hjust=1),
  	axis.text.y = element_blank(),
  	axis.title.y = element_blank(),
  	axis.ticks.y = element_blank(),
  	legend.position="none")

dendroplot <- 
	ggdendrogram(
		dendro,
		rotate=TRUE)+
	theme(
		axis.text.y = element_text(
			size = 5,
			vjust = 0.5,
			hjust = 1.0))

dendrotile <- ggarrange(
	longtile,
	dendroplot,
	labels=c("A","B"),
	ncol=2,
	nrow=1)
#	align="hv")

ggsave("dendrotile.pdf",
	dendrotile,
	width=2000,
	height= 12000, 
	units="px", 
	useDingbats=FALSE)

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
