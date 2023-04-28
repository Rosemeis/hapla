"""
hapla.
Perform optimal window splitting based on LD.
"""

__author__ = "Jonas Meisner"

# Libraries
import os

##### hapla split #####
def main(args):
	print("hapla split by Jonas Meisner (v0.1)")
	print(f"Using {args.threads} thread(s).")
	print("Cite original paper: https://doi.org/10.1093/bioinformatics/btab519")

	# Check input
	assert args.vcf is not None, \
		"Please provide phased genotype file (--bcf or --vcf)!"

	# Control threads of external numerical libraries
	os.environ["MKL_NUM_THREADS"] = str(args.threads)
	os.environ["OMP_NUM_THREADS"] = str(args.threads)
	os.environ["NUMEXPR_NUM_THREADS"] = str(args.threads)
	os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)

	# Import numerical libraries and cython functions
	import numpy as np
	from cyvcf2 import VCF
	from math import ceil
	from hapla import reader_cy
	from hapla import shared_cy

	### Load data into 1-bit matrix
	print("\rLoading VCF/BCF file...", end="")
	v_file = VCF(args.vcf, threads=args.threads)
	n = 2*len(v_file.samples)
	B = ceil(n/8)
	Gt = reader_cy.readVCF(v_file, n//2, B)
	del v_file
	m = Gt.shape[0]
	print(f"\rLoaded phased genotype data: {n} haplotypes and {m} SNPs.")

	### Setup parameters
	assert args.max_length <= m, "Max length > chromosome!"
	maxW = min(args.max_windows, ceil(m/args.min_length) + 1)

	### Optimal window splitting
	# Estimating L matrix
	print("Estimating correlations and L matrix.")
	F = np.zeros(m, dtype=float) # Means
	S = np.zeros(m, dtype=float) # Standard deviations
	L = np.zeros((m, args.max_length), dtype=np.float32)
	shared_cy.estimateL(Gt, F, S, L, args.threshold, n, args.threads)
	del Gt, F, S

	# Estimating E matrix
	print("Estimating E matrix.")
	E = np.zeros((m, args.max_length), dtype=np.float32)
	shared_cy.estimateE(L, E)
	del L

	# Compute optimal path of splits
	print("Estimating cost of paths.")
	C = np.zeros((maxW, m), dtype=np.float32)
	I = np.zeros((maxW, m), dtype=np.int32)
	C.fill(np.inf)
	I.fill(-1)
	shared_cy.estimateC(E, C, I, args.min_length, args.threads)
	del E

	# Reconstruct most optimal path
	print("Reconstructing optimal path.")
	P = np.zeros(maxW, dtype=np.int32)
	optK = maxW-np.argmin(C[:,0][::-1])-1
	del C
	shared_cy.reconstructPath(I, P, optK)
	P = P[:(P.shape[0]-(np.sum(P == 0)-1))]
	if P[-1] == -1:
		P = np.array([m,0], dtype=np.int32)
	print(f"{P.shape[0]-1} optimal blocks.")

	# Save matrices
	np.savetxt(f"{args.out}.windows", P[::-1], fmt="%i")
	print(f"Saved optimal window indices in {args.out}.windows")

