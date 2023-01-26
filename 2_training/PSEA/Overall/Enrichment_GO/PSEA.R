#!/usr/bin/env Rscript
library(clusterProfiler)
data(geneList, package="DOSE")
de <- names(geneList)[abs(geneList) > 2]
ego <- enrichGO(de, OrgDb = "org.Hs.eg.db", ont="BP", readable=TRUE)

library(enrichplot)
goplot(ego)

BH
RcppEigen
ggforce
ggtree
enrichplot
scatterpie
igraph

