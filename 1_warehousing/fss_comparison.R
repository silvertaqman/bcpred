#!/usr/bin/env Rscript
library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(eulerr)
library(pilot)
library(chorddiag)
library(igraph)
library(ggraph)
library(htmlwidgets)
library(manipulateWidget)
library(htmltools)

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

df %>%
	count(Method, Coincidence) %>%
	mutate(
		Method = factor(Method, levels=a$Method),
		Coincidence = factor(Coincidence, levels=b$Coincidence)) %>%
	ggplot(aes(Method, Coincidence, fill= n)) + 
  geom_tile()+
  theme(
  	axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1),
  	axis.text.y=element_text(size=5))

ggsave("tile.pdf", 
	width=1500,
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

# Dendrogram
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
# Otra opcion

d %>%
	hclust() %>%
	as.dendrogram() %>%
	ggraph("dendogram")+
		geom_edge_elbow()

# Chord Diagram
# circlize::chordDiagram(scores, transparency = 0.5) # doesn't work for +360 features , so first 360 features are selected

order <- levels(b$Coincidence)[1:50]

df %>%
	count(Method, Coincidence) %>%
	mutate(
		Method = factor(Method, levels=a$Method),
		Coincidence = factor(Coincidence, levels=b$Coincidence)) %>%
		filter(Coincidence %in% order) %>% # %>% barplot(Coincidence)
		graph_from_data_frame() %>%
		ggraph(aes(colour=Method),layout="linear", circular=TRUE)+
			geom_edge_arc(
				edge_colour="black",
				edge_alpha=0.3, 
				edge_width=0.2)+
			geom_node_point(
				color="#69b3a2", 
				size=5) +
			geom_node_text(
				aes(label=name),
				repel = TRUE, 
				size=8, 
				color="#69b3a2") +
		theme_pilot()+
		theme(
		  legend.position="none",
		  plot.margin=unit(rep(2,4), "cm"))
  


# Sankey Diagram
links <- df %>% 
	count(Method, Coincidence) %>%
	mutate(
		Method = factor(Method, levels=a$Method),
		Coincidence = factor(Coincidence, levels=b$Coincidence))
		
nodes <- data.frame(
  name=c(
  	as.character(links$Method),
  	as.character(links$Coincidence)) %>% 
    unique()
  )

links$IDMethod <- match(links$Method, nodes$name)-1 
links$IDCoincidence <- match(links$Coincidence, nodes$name)-1

sn<-sankeyNetwork(
	Links = links,
	Nodes = nodes,
	Source = "IDMethod",
	Target = "IDCoincidence",
	Value = "n",
	NodeID = "name", 
	sinksRight=FALSE,
	fontSize= 20,
	nodeWidth = 20,
	margin = list(left = 50),
	colourScale = JS("d3.scaleOrdinal(d3.schemeCategory20b);"))

onRender(
  sn,
  '
  function(el, x) {
    d3.selectAll(".node text").attr("text-anchor", "begin").attr("x", 20);
  }
  '
)

