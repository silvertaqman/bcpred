#!/bin/bash
# A function to use use Spencer and generate a prediction file
# Paste your fasta proteins at input.fasta file
Rscript prot2descriptors.R input.fasta descriptors.csv
pigz descriptors.csv --force
python3 descriptor2status.py
