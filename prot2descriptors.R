#!/usr/bin/env Rscript
# IN: take sequences from a TOF sample in fasta format
# OUT: A dataframe with frequencies for AA as a CSV
library(Rcpi)
library(readr)
library(dplyr)
library(tidyr)
# Return all descriptors for a certain sequence
extractor <- function(x) c(extractProtAAC(x),extractProtDC(x),extractProtTC(x),extractProtAPAAC(x),extractProtMoreauBroto(x))

# load input sequences
df <- readFASTA("input.fasta") |>
	as_tibble() |>
	pivot_longer(everything(), names_to="Protein", values_to="Sequence")

# calculate descriptors for every seq and bind
first <- extractor(df$Sequence[1])
first <- data.frame(feature = names(first))
for(i in df$Sequence) first = bind_cols(first, extractor(i))
descriptors <- t(first[,-1])
colnames(descriptors) <- first$feature
descriptors <- as_tibble(descriptors)
# export to csv (with names)
df |>
	bind_cols(descriptors)|>
	write_csv("descriptors.csv")
