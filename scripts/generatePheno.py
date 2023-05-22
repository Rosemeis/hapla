"""
Generate continuous phenotypes for simulated data.
Sample causal effect sizes.

Usage:
python3 generatePheno.py --bcf file.bcf --causal 10 --seed 1 --threads 64 \
	--out output.prefix
"""

__author__ = "Jonas Meisner"

# Libraries
import argparse
import os

### Argparse
parser = argparse.ArgumentParser()
parser.add_argument("-g", "--vcf", "--bcf",
	help="Genotype file in VCF/BCF format")
parser.add_argument("-f", "--filelist",
	help="Filelist with paths to haplotype cluster files")
parser.add_argument("-z", "--clusters",
	help="Path to a single haplotype cluster assignment file")
parser.add_argument("-k", "--num_clusters",
	help="Path to either single path or filelist of haplotype cluster counts")
parser.add_argument("-b", "--beta",
	help="Path to pre-estimated betas from training data")
parser.add_argument("-c", "--causal", type=int, default=100,
	help="Number of causal SNPs (100)")
parser.add_argument("-e", "--h2", type=int, default=5,
	help="Heritability of trait as integer (5 = 0.5)")
parser.add_argument("-s", "--seed", type=int, default=42,
	help="Set random seed (42)")
parser.add_argument("-a", "--alpha", type=float, default=-1.0,
	help="Negative selection parameter (-1.0)")
parser.add_argument("-t", "--threads", type=int, default=1,
	help="Number of threads (1)")
parser.add_argument("-o", "--out", default="hapla.generate",
	help="Prefix for output files")
parser.add_argument("--save_beta", action="store_true",
	help="Save the sampled causal betas")
parser.add_argument("--save_regenie", action="store_true",
	help="Save extra phenotype file in regenie format")
args = parser.parse_args()

# Check input
assert args.vcf is not None, "Please provide genotype file (--bcf or --vcf)!"
if args.beta is None:
	assert args.causal is not None, "Please provide number of casual SNPs (--causal)!"
if args.save_regenie:
	assert args.vcf is not None, "VCF/BCF file is needed for sample list!"

# Control threads of external numerical libraries
os.environ["MKL_NUM_THREADS"] = str(args.threads)
os.environ["OMP_NUM_THREADS"] = str(args.threads)
os.environ["NUMEXPR_NUM_THREADS"] = str(args.threads)
os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)

# Import numerical libraries
import numpy as np
from cyvcf2 import VCF
from math import ceil, sqrt
from hapla import reader_cy

### Load data
if args.vcf is not None:
	print("\rLoading VCF/BCF file...", end="")
	v_file = VCF(args.vcf, threads=args.threads)
	n = len(v_file.samples)
	if args.save_regenie:
		s_list = v_file.samples
	if (args.clusters is None) and (args.filelist is None):
		B = ceil(2*n/8)
		Gt = reader_cy.readVCF(v_file, n, B)
		m = Gt.shape[0]
		print(f"\rLoaded genotype data: {n} samples and {m} SNPs.")
	del v_file
if (args.clusters is not None) or (args.filelist is not None):
	if args.filelist is not None:
		# Haplotype clusters
		Z_list = []
		with open(args.filelist) as f:
			file_c = 1
			for chr in f:
				Z_list.append(np.load(chr.strip("\n")))
				print(f"\rParsed file #{file_c}", end="")
				file_c += 1
		Z_mat = np.concatenate(Z_list, axis=0)
		del Z_list

		# Files with number of clusters
		K_list = []
		with open(args.num_clusters) as f:
			file_c = 1
			for chr in f:
				K_list.append(np.loadtxt(chr.strip("\n"), dtype=np.uint8))
				file_c += 1
		K_vec = np.concatenate(K_list, dtype=np.uint8)
		del K_list
	else:
		Z_mat = np.load(args.clusters)
		K_vec = np.loadtxt(args.num_clusters, dtype=np.uint8)
	print("\rLoaded haplotype cluster assignments of " + \
		f"{Z_mat.shape[1]} haplotypes in {Z_mat.shape[0]} windows.")
	n = Z_mat.shape[1]//2
	m = np.sum(K_vec)

### Causal betas and sampling
assert (args.h2 > 0) and (args.h2 < 10), "Invalid value for h2!" 
h2 = float(f"0.{args.h2}")
if args.beta is None:
	G = np.zeros((args.causal, n), dtype=np.uint8) # Genotypes or haplotype clusters
	np.random.seed(args.seed) # Set random seed
	p = np.sort(np.random.permutation(m)[:args.causal]).astype(int) # Select causals
	if (args.clusters is not None) or (args.filelist is not None):
		C_vec = np.arange(m, dtype=int)[p]
		reader_cy.convertHaplo(Z_mat, G, K_vec, C_vec)
		del Z_mat, K_vec, C_vec
	else:
		reader_cy.genotypeBit(Gt, G, p)
		del Gt
	f = np.mean(G, axis=1)/2.0
	b = np.random.normal(loc=0.0, scale=(f*(1.0-f))**args.alpha, size=args.causal)
	B = np.zeros(m)
	B[p] = b
else:
	beta = np.loadtxt(args.beta, dtype=float)
	p = np.arange(m, dtype=int)[beta != 0]
	b = beta[p]
	del beta
	G = np.zeros((p.shape[0], n), dtype=np.uint8) # Genotypes or haplotype clusters
	if (args.clusters is not None) or (args.filelist is not None):
		C_vec = np.arange(m, dtype=int)[p]
		reader_cy.convertHaplo(Z_mat, G, K_vec, C_vec)
		del Z_mat, K_vec, C_vec
	else:
		reader_cy.genotypeBit(Gt, G, p)
		del Gt

### Estimate phenotypes
# Genetic contribution
X = np.dot(G.T, b)
X -= np.mean(X)
G_scal = sqrt(h2)/(np.linalg.norm(X)/np.sqrt(n))
G_liab = X*G_scal

# Environmental contribution
E = np.random.normal(loc=0.0, scale=sqrt(1 - h2), size=n)
E -= np.mean(E)
E_liab = E*(sqrt(1.0 - h2)/(np.linalg.norm(E)/np.sqrt(n)))

# Generate phenotype
Y = G_liab + E_liab

### Save output
np.savetxt(f"{args.out}.pheno", Y, fmt="%.7f")
print(f"Saved continuous phenotypes as {args.out}.pheno")
np.savetxt(f"{args.out}.prs", G_liab, fmt="%.7f")
print(f"Saved PRS as {args.out}.prs")
if args.save_regenie:
	Y_regenie = np.repeat(np.array(s_list), 2).reshape(n, 2)
	Y_regenie = np.hstack((Y_regenie, np.round(Y.reshape(-1,1), 7)))
	np.savetxt(f"{args.out}.regenie.pheno", Y_regenie, delimiter="\t", \
	    comments="", header="FID\tIID\tY1", fmt="%s")
	print("Saved continuous phenotypes in regenie format as " + \
	f"{args.out}.regenie.pheno")
if (args.save_beta) and (args.beta is None):
	np.savetxt(f"{args.out}.beta", B, fmt="%.7f")
	print(f"Saved causal betas as {args.out}.beta")
