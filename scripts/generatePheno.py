"""
Generate continuous phenotypes for simulated data.
Sample causal effect sizes.

Usage:
python3 generatePheno.py --vcf file.bcf --windows file.windows \
	--causal 10 --seed 1 --threads 8 --out output.prefix
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
	help="Filelist with paths to multiple haplotype cluster assignment files")
parser.add_argument("-z", "--clusters",
	help="Path to a single haplotype cluster assignment file")
parser.add_argument("-c", "--causal", type=int,
	help="Number of causal SNPs")
parser.add_argument("-e", "--h2", type=int, default=5,
	help="Heritability of trait as integer (0.x)")
parser.add_argument("-s", "--seed", type=int, default=42,
	help="Set random seed (42)")
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
from math import ceil, sqrt
from cyvcf2 import VCF
from hapla import reader_cy, shared_cy

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
		Z_list = []
		with open(args.filelist) as f:
			file_c = 1
			for chr in f:
				Z_list.append(np.load(chr.strip("\n")))
				print(f"\rParsed file #{file_c}", end="")
				file_c += 1
		Z_mat = np.concatenate(Z_list, axis=0)
		del Z_list
	else:
		Z_mat = np.load(args.clusters)
	print("\rLoaded haplotype cluster assignments of " + \
		f"{Z_mat.shape[1]} haplotypes in {Z_mat.shape[0]} windows.")
	n = Z_mat.shape[1]//2

	# Estimate total number of haplotype cluster assignments and frequencies
	K_vec = np.max(Z_mat, axis=1) + 1
	m = np.sum(K_vec)
	pi = np.zeros(m, dtype=float)
	reader_cy.estimateFreqs(Z_mat, K_vec, pi)
	C_vec = np.arange(m, dtype=int)[(pi > 0.01) and (pi < 0.99)]
	m = C_vec.shape[0]

### Causal betas and sampling
assert (args.h2 > 0) and (args.h2 < 10), "Invalid value for h2!" 
h2 = float(f"0.{args.h2}")
G = np.zeros((args.causal, n), dtype=np.uint8) # Genotypes
np.random.seed(args.seed) # Set random seed
p = np.sort(np.random.permutation(m)[:args.causal]).astype(int) # Select causals
if (args.clusters is not None) or (args.filelist is not None):
	C_vec = C_vec[p]
	reader_cy.convertHaplo(Z_mat, G, K_vec, C_vec)
	del Z_mat, K_vec, C_vec
else:
	reader_cy.genotypeBit(Gt, G, p)
	del Gt

b = np.random.normal(loc=0.0, scale=sqrt(h2/args.causal), size=args.causal)
B = np.zeros(m)
B[p] = b

### Estimate phenotypes
# Genetic contribution
X = np.dot(G.T, b)
X -= np.mean(X)
X /= (np.linalg.norm(X)/np.sqrt(n))
G_liab = sqrt(h2)*X

# Environmental contribution
E = np.random.normal(loc=0.0, scale=sqrt(1 - h2), size=n)
E -= np.mean(E)
E /= (np.linalg.norm(E)/np.sqrt(n))
E_liab = sqrt(1.0 - h2)*E

# Generate phenotype
Y = G_liab + E_liab

### Save output
np.savetxt(f"{args.out}.h{args.h2}.s{args.seed}.c{args.causal}.pheno", \
	Y, fmt="%.7f")
print("Saved continuous phenotypes as " + \
	f"{args.out}.h{args.h2}.s{args.seed}.c{args.causal}.pheno")
np.savetxt(f"{args.out}.h{args.h2}.s{args.seed}.c{args.causal}.prs", \
	G_liab, fmt="%.7f")
print(f"Saved PRS as {args.out}.h{args.h2}.s{args.seed}.c{args.causal}.prs")
if args.save_regenie:
	Y_regenie = np.repeat(np.array(s_list), 2).reshape(n, 2)
	Y_regenie = np.hstack((Y_regenie, np.round(Y.reshape(-1,1), 7)))
	np.savetxt(f"{args.out}.h{args.h2}.s{args.seed}.c{args.causal}.regenie.pheno", \
		Y_regenie, delimiter="\t", comments="", header="FID\tIID\tY1", \
		fmt="%s")
	print("Saved continuous phenotypes in regenie format as " + \
	f"{args.out}.h{args.h2}.s{args.seed}.c{args.causal}.regenie.pheno")
