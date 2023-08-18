"""
Generate continuous phenotypes for simulated data.
Sample causal effect sizes.

Usage:
python3 generatePheno.py --bcf file.bcf --causal 100 --seed 1 --threads 16 \
	--out output.prefix
"""

__author__ = "Jonas Meisner"

# Libraries
import argparse
import os
import subprocess

def extract_length(filename):
	process = subprocess.Popen(['wc', '-l', filename], stdout=subprocess.PIPE)
	result, _ = process.communicate()
	return int(result.split()[0])

### Argparse
parser = argparse.ArgumentParser()
parser.add_argument("-g", "--vcf", "--bcf",
	help="Genotype file in VCF/BCF format")
parser.add_argument("-b", "--bfile",
	help="Genotype file in binary PLINK format")
parser.add_argument("-f", "--filelist",
	help="Filelist with paths to haplotype cluster files")
parser.add_argument("-c", "--causal", type=int, default=1000,
	help="Number of causal SNPs (1000)")
parser.add_argument("-e", "--h2", type=int, default=5,
	help="Heritability of trait as integer (5 = 0.5)")
parser.add_argument("-s", "--seed", type=int, default=42,
	help="Set random seed (42)")
parser.add_argument("-t", "--threads", type=int, default=1,
	help="Number of threads (1)")
parser.add_argument("-o", "--out", default="pheno.generate",
	help="Prefix for output files")
parser.add_argument("--save_beta", action="store_true",
	help="Save the sampled causal betas")
parser.add_argument("--save_plink", action="store_true",
	help="Save extra phenotype file in PLINK format")
args = parser.parse_args()

# Check input
if args.filelist is None:
	assert (args.vcf is not None) or (args.bfile is not None), \
		"Please provide genotype file (--bcf, --vcf or --bfile)!"
if args.save_plink:
	assert (args.vcf is not None) or (args.bfile is not None), \
		"VCF/BCF or PLINK files are needed for sample list!"
assert (args.h2 > 0) and (args.h2 < 10), "Invalid value for h2!"

# Control threads of external numerical libraries
os.environ["MKL_NUM_THREADS"] = str(args.threads)
os.environ["OMP_NUM_THREADS"] = str(args.threads)
os.environ["NUMEXPR_NUM_THREADS"] = str(args.threads)
os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)

# Import numerical libraries
import numpy as np
from cyvcf2 import VCF
from math import ceil, sqrt
from src import reader_cy

### Load data
if args.vcf is not None: # VCF/BCF file
	print("\rLoading VCF/BCF file...", end="")
	v_file = VCF(args.vcf, threads=args.threads)
	n = len(v_file.samples)
	if args.save_plink:
		s_list = np.array([v_file.samples]).reshape(-1,1)
	if args.filelist is None:
		B = ceil(2*n/8)
		G_mat = reader_cy.readVCF(v_file, n, B)
		m = G_mat.shape[0]
		print(f"\rLoaded genotype data: {n} samples and {m} SNPs.")
	del v_file
elif args.bfile is not None: # Binary PLINK files
	print("\rLoading binary PLINK files...", end="")
	
	# Finding length of .fam and .bim file
	n = extract_length(f"{args.bfile}.fam")
	m = extract_length(f"{args.bfile}.bim")

	# Read .bed file
	with open(f"{args.bfile}.bed", "rb") as bed:
		G_mat = np.fromfile(bed, dtype=np.uint8, offset=3)
	B = ceil(n/4)
	G_mat.shape = (m, B)
	print(f"\rLoaded genotype data: {n} samples and {m} SNPs.")
if args.filelist is not None: # Haplotype clusters
	Z_list = []
	with open(args.filelist) as f:
		file_c = 1
		for chr in f:
			Z_list.append(np.load(chr.strip("\n")))
			print(f"\rParsed file #{file_c}", end="")
			file_c += 1
	Z_mat = np.concatenate(Z_list, axis=0)
	del Z_list
	print("\rLoaded haplotype cluster alleles of " + \
		f"{Z_mat.shape[1]} haplotypes in {Z_mat.shape[0]} windows.")
	W = Z_mat.shape[0]
	n = Z_mat.shape[1]//2

	# Estimate total number of haplotype cluster alleles
	K_vec = np.max(Z_mat, axis=1) # Dummy encoding
	m = np.sum(K_vec, dtype=int)

### Causal betas and sampling
h2 = float(f"0.{args.h2}")
G = np.zeros((args.causal, n), dtype=float) # Genotypes or haplotype clusters
np.random.seed(args.seed) # Set random seed
p = np.sort(np.random.permutation(m)[:args.causal]).astype(int) # Select causal loci
if args.vcf is not None: # VCF/BCF
	reader_cy.phenoVCF(G_mat, G, p)
	del G_mat
elif args.bfile is not None: # binary PLINK
	reader_cy.phenoPlink(G_mat, G, p)
	del G_mat
else: # Haplotype cluster alleles
	C_vec = np.arange(m, dtype=int)[p]
	reader_cy.phenoHaplo(Z_mat, G, K_vec, C_vec)
	del Z_mat, K_vec, C_vec

# Sample causal effects
b = np.random.normal(loc=0.0, scale=1.0, size=args.causal)
B = np.zeros(m)
B[p] = b

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
np.savetxt(f"{args.out}.set", p, fmt="%i")
print(f"Saved causal SNP set as {args.out}.set")
if args.save_plink:
	if args.bfile:
		Y_plink = np.hstack((f_list, s_list))
	else:
		Y_plink = s_list.repeat(2, axis=1)
	Y_plink = np.hstack((Y_plink, np.round(Y.reshape(-1,1), 7)))
	np.savetxt(f"{args.out}.plink.pheno", Y_plink, delimiter="\t", fmt="%s")
	print("Saved continuous phenotypes in plink format as " + \
	f"{args.out}.plink.pheno")
if args.save_beta:
	np.savetxt(f"{args.out}.beta", B*G_scal, fmt="%.7f")
	print(f"Saved causal betas as {args.out}.beta")
