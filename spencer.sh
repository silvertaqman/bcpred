#!/bin/bash
# A function to use use Spencer and generate a prediction file
# Paste your fasta proteins at input.fasta file
Rscript prot2descriptors.R
pigz descriptors.csv --overwrite -y
python3 descriptor2status.py
