#!/usr/bin/env Rscript
library(readr)
library(tidyverse)
library(Rcpi)
# Load data
a <- readr::read_csv("topfeatures.csv")
?extractProtAPAAC
# Show physicochemical properties (APPAC)
a |> 
	mutate(n=nchar(Coincidence)) |> 
	filter(n>4)
# Show Compositional properties
a |> 
	mutate(n=nchar(Coincidence)) |> 
	filter(n<4)
# Show which aa have the greatest frequencies
a |> 
	mutate(n=nchar(Coincidence)) |> 
	filter(n<4) |>
	separate_longer_position(width=1, Coincidence) |>
	count(Coincidence, sort=TRUE)
