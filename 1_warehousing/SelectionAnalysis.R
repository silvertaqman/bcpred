#!/usr/bin/env Rscript
library(readr)
library(tidyverse)
################################################################
args = commandArgs(trailingOnly=TRUE)
df <- read_csv("./selection.csv.gz") %>%
	select(!...1) %>%
	pivot_longer(
		everything(),
		names_to = "Method",
		values_to = "Coincidence") %>%
	drop_na(Coincidence) %>%
    mutate_if(is.character, as.factor) %>%
    count(Coincidence) %>% 
    arrange(desc(n)) %>%
    select(Coincidence)

write_csv(df[1:args[1], ], "topfeatures.csv")
# Use any number of features as input
# Get variance plot
