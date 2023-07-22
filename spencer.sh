#!/bin/bash
# A function to use use Spencer and generate a prediction file
# Paste your fasta proteins at input.fasta file
Rscript prot2descriptors.R input.fasta descriptors.csv
<<<<<<< HEAD
pigz descriptors.csv --force
=======
pigz descriptors.csv --overwrite -y
>>>>>>> 04b0460809710406095b4e1a427dc0fcdc9198cd
python3 descriptor2status.py
