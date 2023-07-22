#!/usr/bin/env Rscript
library(readr)
library(ggpubr)
library(purrr)
library(pilot)
library(tidyverse)
#library(e1071)
library(pilot)
library(knitr)
library(grid)
library(cowplot)
<<<<<<< HEAD
library(igraph)
library(ggraph)
library(ggdendro)
library(dendextend)
library(patchwork)
library(factoextra)
library(zoo)
=======
#library(igraph)
#library(ggraph)
#library(ggdendro)
#library(dendextend)
library(patchwork)
>>>>>>> 04b0460809710406095b4e1a427dc0fcdc9198cd
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
	rename(Class = V2) %>%
	mutate(
		group = rep('Original', 3286240),
		property = pmap(., 
			~ifelse(
				nchar(..2) > 3, # only frequences have this kind of names
				'Atributo',
				'Composición')) %>% 
				unlist) %>% 
<<<<<<< HEAD
	select(group,property, Class)
=======
	select(group,property, Class) 
>>>>>>> 04b0460809710406095b4e1a427dc0fcdc9198cd

mixbal <- read_csv("../2_training/Mix_BC_srbal.csv.gz") %>%
	pivot_longer(
		!Class,
		names_to = "aminoacidseq",
		values_to = "frequence") %>%
	mutate(
<<<<<<< HEAD
		group = rep('Balanceado', 128616),
		property = pmap(.,
			~ifelse(nchar(..2) > 3,'Atributo','Composición')) %>%
			unlist) %>%
=======
		group = rep('Balanceado', 139800),
		property = pmap(
			.,
			~ifelse(nchar(..2) <= 3,'Composición','Atributo')) %>%
				unlist) %>%
>>>>>>> 04b0460809710406095b4e1a427dc0fcdc9198cd
	select(group,property, Class)

# Merge datasets

mix <- mix %>%
	bind_rows(mixbal) %>%
	mutate(
		Class = factor(Class, labels= c("Control","Paciente")))
rm(mixbal)

# Data barplot (before)

warehouse <- mix %>%
	count(group, property, Class) %>%
	mutate(group = factor(group, levels = c("Original","Balanceado"))) %>%
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
  	scale_fill_pilot()+
  	scale_y_sqrt()

# Merge two plots
ggsave("balance.png", 
	warehouse,
	width = 3200, 
	height = 1600,
	units = "px")
<<<<<<< HEAD
#####################################################################
=======

>>>>>>> 04b0460809710406095b4e1a427dc0fcdc9198cd
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

<<<<<<< HEAD
scores <- scores[which(rowSums(scores)>4),]

# export scores with > 7 frequences
=======
scores <- scores[rowSums(scores)>5,]

# export scores with > 5 frequences
>>>>>>> 04b0460809710406095b4e1a427dc0fcdc9198cd
read_csv("Mix_BC.csv.gz") %>%
	select(any_of(rownames(scores))) %>%
	write_csv("Mix_BC_selected.csv")

<<<<<<< HEAD
d = dist(scores, method = "binary")
cut <- 4  # Number of clusters
hc = hclust(d, method="ward.D")
dendr <- dendro_data(hc, type = "rectangle") 
clust <- cutree(hc, k = cut)               # find 'cut' clusters
clust.df <- data.frame(label = names(clust), cluster = clust)

# Split dendrogram into upper grey section and lower coloured section
height <- unique(dendr$segments$y)[order(unique(dendr$segments$y), decreasing = TRUE)]
cut.height <- mean(c(height[cut], height[cut-1]))
dendr$segments$line <- ifelse(dendr$segments$y == dendr$segments$yend &
   dendr$segments$y > cut.height, 1, 2)
dendr$segments$line <- ifelse(dendr$segments$yend  > cut.height, 1, dendr$segments$line)

# Number the clusters
dendr$segments$cluster <- c(-1, diff(dendr$segments$line))
change <- which(dendr$segments$cluster == 1)
for (i in 1:cut) dendr$segments$cluster[change[i]] = i + 1
dendr$segments$cluster <-  ifelse(dendr$segments$line == 1, 1, 
             ifelse(dendr$segments$cluster == 0, NA, dendr$segments$cluster))
dendr$segments$cluster <- na.locf(dendr$segments$cluster) 

# Consistent numbering between segment$cluster and label$cluster
clust.df$label <- factor(clust.df$label, levels = levels(dendr$labels$label))
clust.df <- arrange(clust.df, label)
clust.df$cluster <- factor((clust.df$cluster), levels = unique(clust.df$cluster), labels = (1:cut) + 1)
dendr[["labels"]] <- merge(dendr[["labels"]], clust.df, by = "label")

# Positions for cluster labels
n.rle <- rle(dendr$segments$cluster)
N <- cumsum(n.rle$lengths)
N <- N[seq(1, length(N), 2)] + 1
N.df <- dendr$segments[N, ]
N.df$cluster <- N.df$cluster - 1

# clusterize features 

dd <- read_csv("Mix_BC_sr.csv.gz") %>% 
        select(!c(...1, Class)) 

k2 <- kmeans(t(dd), centers = 2, nstart = 25)
p <- fviz_cluster(k2, data = t(dd))

names <- p$data$name
cluster <- p$data$cluster

# set a value if get a coincidence ?mutate
=======
# dendrogram

d = dist(scores, method = "binary")
hc = hclust(d, method="ward.D")
dendro <- as.dendrogram(hc)
>>>>>>> 04b0460809710406095b4e1a427dc0fcdc9198cd

tileplot <- df %>%
	count(Method, Coincidence) %>%
	mutate(
#		Method = factor(Method, levels=a$Method),
<<<<<<< HEAD
		Coincidence = factor(
			Coincidence, 
			levels=partition_leaves(as.dendrogram(hc))[[1]])) %>%
	group_by(n) %>%
	mutate(Class = ifelse(
		Coincidence %in% names[which(cluster==2)],
		"Positivo",
		"Negativo")) %>%
	drop_na() %>%
	ggplot(
		aes(Method,Coincidence, 
			fill = n,
			alpha = ifelse(Class=="Negativo",0.85,NA))) + 
  geom_tile()+
  theme(
  	axis.text.x=element_text(angle=90,size=9),
  	axis.text.y=element_text(size=8),
  	legend.position="none")+
  labs(x="Métodos", y="")

dendroplot <- ggplot() + 
   geom_segment(data = segment(dendr), 
      aes(x=x, y=y, xend=xend, yend=yend, size=factor(line), colour=factor(cluster)), 
      lineend = "square", show.legend = FALSE) + 
   scale_color_pilot() +
   scale_size_manual(values = c(.1, 1))+
   labs(x = NULL, y = NULL) +
   coord_flip() +
    theme(
    	axis.text.x = element_text(angle=90,color = "gray12", size = 12),
    	axis.line.y = element_blank(),
        axis.ticks.y = element_blank(),
        axis.text.y = element_blank(),
        axis.title.y = element_blank(),
        panel.background = element_rect(fill = "white"),
        panel.grid = element_blank())
=======
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
>>>>>>> 04b0460809710406095b4e1a427dc0fcdc9198cd

fss_comparison <- plot_grid(
	tileplot, 
	dendroplot, 
	rel_widths=c(3,1),
	rel_heights=c(2,2),
	align = "h",
	scale=c(1,1.066)
)
<<<<<<< HEAD
	
ggsave("fss.png",
	fss_comparison, 
	device = "png",
	dpi=300,
	width = 1800, 
	height = 2200,
	bg = "white", 
	units = "px")
=======
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
>>>>>>> 04b0460809710406095b4e1a427dc0fcdc9198cd

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
<<<<<<< HEAD
 
 # Frequencies
 # dendrogram

tabla <- b %>% 
	select(!n) %>%
	transmute(Co = as.character(Coincidence)) %>%
	filter(nchar(Co) < 4) %>%
	separate(Co, c("rem", "A", "AA", "AAA"), "") %>%
	select(!rem) %>%
	pivot_longer(everything(), names_to=NULL, values_to="amino") %>%
	drop_na(amino) %>%
	group_by(amino) %>%
	table()

barplot(tabla)
shapiro.test(tabla)
names(tabla[which(tabla > mean(tabla)+qt(0.99, 19)*sd(tabla)/sqrt(20))])
names(tabla[which(tabla < mean(tabla)-qt(0.99, 19)*sd(tabla)/sqrt(20))])

# Codons
codones <- data.frame(codon=c("CAA","CAG", "AAA", "AAG", "UUA","UUG","CUU","CUC","CUA","CUG","CCU","CAC","CAA","CAG","GUU","GUC","GUA","GUG"), amino=c("G","G","K","K","L","L","L","L","L","L","P","P","P","P","V","V","V","V"))
=======
>>>>>>> 04b0460809710406095b4e1a427dc0fcdc9198cd
