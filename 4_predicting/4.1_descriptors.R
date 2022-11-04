#!/usr/bin/env Rscript
library(Rcpi)
library(readr)
library(dplyr)
read_fasta("input.csv") %>%
	bind_cols(
		across(V3, extractProtAAC)
	)
	
df <- readFASTA('input.fasta') %>% 
