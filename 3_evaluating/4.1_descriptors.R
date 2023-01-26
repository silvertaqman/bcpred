#!/usr/bin/env Rscript
# IN: take sequences from a TOF sample in fasta format
# OUT: A dataframe with frequencies for AA as a CSV
library(Rcpi)
library(readr)
library(dplyr)
read_fasta("input.csv") %>%
	bind_cols(
		across(V3, extractProtAAC)
	)
	
df <- readFASTA('input.fasta') %>% 
