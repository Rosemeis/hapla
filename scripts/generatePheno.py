"""
Generate phenotypes from genotype data in binary PLINK format.

Usage:
python3 generatePheno.py --bfile example --causal 1000 --phenos 10 --out output example
"""

__author__ = "Jonas Meisner"

# Libraries
import argparse
import os

### Argparse
parser = argparse.ArgumentParser()
parser.add_argument("-b", "--bfile",
	help="Genotype file in binary PLINK format")
parser.add_argument("-e", "--h2", type=float, default=0.5,
	help="Heritability of simulated traits (0.5)")
parser.add_argument("-c", "--causal", type=int, default=1000,
	help="Number of causal SNPs (1000)")
parser.add_argument("-p", "--phenos", type=int, default=1,
	help="Number of phenotypes to simulate (1)")
parser.add_argument("--binary", action="store_true",
	help="Binary phenotypes from liability threshold model")
parser.add_argument("--prevalence", type=float, default=0.1,
	help="Prevalence of trait (0.1)")
parser.add_argument("-o", "--out", default="pheno.generate",
	help="Prefix for output files")
args = parser.parse_args()

# Import numerical libraries
import numpy as np
from cyvcf2 import VCF
from math import ceil, sqrt
from scipy.stats import norm
from hapla import functions
from hapla import reader_cy

### Load data
print("\rLoading PLINK file...", end="")
	
# Finding length of .fam and .bim file
assert os.path.isfile(f"{args.bfile}.bed"), "bed file doesn't exist!"
assert os.path.isfile(f"{args.bfile}.bim"), "bim file doesn't exist!"
assert os.path.isfile(f"{args.bfile}.fam"), "fam file doesn't exist!"
n = functions.extract_length(f"{args.bfile}.fam")
m = functions.extract_length(f"{args.bfile}.bim")
fam = np.loadtxt(f"{args.bfile}.fam", usecols=[0,1], dtype=np.str_)

# Read .bed file
with open(f"{args.bfile}.bed", "rb") as bed:
	G_mat = np.fromfile(bed, dtype=np.uint8, offset=3)
B = ceil(n/4)
G_mat.shape = (m, B)
print(f"\rLoaded genotype data: {n} samples and {m} SNPs.")

### Simulate phenotypes
Y = np.zeros((n, args.phenos), dtype=float) # Phenotype matrix
Z = np.zeros((n, args.phenos), dtype=float) # Breeding values matrix

# Extract causals
G = np.zeros((args.causal, n), dtype=float) # Genotypes or haplotype clusters

# Sample causal SNPs
for p in range(args.phenos):
	# Sample causal loci
	c = np.sort(np.random.permutation(m)[:G.shape[0]]).astype(int)
	reader_cy.phenoPlink(G_mat, G, c)

	# Sample causal effects and estimate true PGS:
	b = np.random.normal(loc=0.0, scale=sqrt(args.h2/float(G.shape[0])), \
		size=G.shape[0])

	# Genetic contribution
	X = np.dot(G.T, b)
	X_scale = sqrt(args.h2)/np.std(X, ddof=0)
	X *= X_scale
	X -= np.mean(X)

	# Environmental contribution
	E = np.random.normal(loc=0.0, scale=sqrt(1 - args.h2), size=n)
	E_scale = sqrt(1 - args.h2)/np.std(E, ddof=0)
	E *= E_scale
	E -= np.mean(E)

	# Generate phenotype
	Y[:,p] = X + E
	Z[:,p] = X

	# Use liability threshold model
	if args.binary:
		Y[:,p] = Y[:,p] > norm.ppf(1 - args.prevalence)

# Save phenotypes and breeding values
Y = np.hstack((fam, np.round(Y, 7)))
Z = np.hstack((fam, np.round(Z, 7)))
np.savetxt(f"{args.out}.pheno", Y, fmt="%s", delimiter=" ")
np.savetxt(f"{args.out}.breed", Z, fmt="%s", delimiter=" ")
print(f"Saved simulated phenotypes as {args.out}.pheno")
print(f"Saved simulated breeding values as {args.out}.breed")
