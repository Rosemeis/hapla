"""
hapla.
Optimal window splits based on LD.
"""

__author__ = "Jonas Meisner"

# Libraries
import os
from time import time

##### hapla cluster #####
def main(args):
	print("-----------------------------------")
	print("hapla by Jonas Meisner (v0.7)")
	print(f"hapla split using {args.threads} thread(s)")
	print("-----------------------------------\n")

	# Check input
	assert args.vcf is not None, \
		"No phased genotype file (--bcf or --vcf)!"
	assert args.min_length > 1, "Minimum length needs to > 1!"
	assert args.min_length <= args.max_length, "Minimum length > maximum length!"
	assert args.batch >= args.max_length, "Not a valid batch size!"
	start = time()

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
	from hapla import split_cy

	# Load data into 1-bit matrix
	print("\rLoading VCF/BCF file...", end="")
	v_file = VCF(args.vcf, threads=args.threads)
	m = 0
	n = 2*len(v_file.samples)
	B = ceil(n/8)

	# Check number of sites and allocate memory
	for variant in v_file:
		m += 1
	assert args.max_length <= m, "Maximum length > chromosome length!"
	G = np.zeros((m, B), dtype=np.uint8)

	# Read variants into matrix
	v_file = VCF(args.vcf, threads=args.threads)
	j = 0
	for variant in v_file:
		V = variant.genotype.array()
		reader_cy.readVar(G, V, j, n//2)
		j += 1
	del V
	t_par = time()-start
	print(f"\rLoaded phased genotype data: {n} haplotypes and {m} SNPs.")

	# Containers
	b = 1
	W = [0]
	m_b = 0
	m_w = ceil(args.batch/args.min_length) + 1
	X = np.zeros((args.batch, n), dtype=np.float32)
	L = np.zeros((args.batch, args.max_length), dtype=np.float32)
	E = np.zeros((args.batch, args.max_length), dtype=np.float32)
	C = np.zeros((args.batch, m_w), dtype=np.float32)
	I = np.zeros((args.batch, m_w), dtype=np.int32)
	P = np.zeros(m_w, dtype=np.int32)

	# Loop through batches
	print("Estimating optimal window splits...")
	while (m_b + args.batch) < (m - args.max_length):
		# Reset arrays
		L.fill(0.0)
		E.fill(0.0)
		C.fill(np.inf)
		I.fill(-1)
		P.fill(0.0)

		# Estimate matrices and compute optimal paths
		split_cy.extractG(G, X, m_b, args.threads)
		split_cy.estimateL(X, L, args.threshold, args.threads)
		split_cy.estimateE(L, E)
		optK = split_cy.estimateC(E, C, I, args.min_length, args.threads)
		split_cy.constructP(I, P, m_b, optK)
		P_w = P[P <= (m_b + args.batch - args.max_length)]
		P_w = P_w[:(P_w.shape[0]-(np.sum(P_w == 0)-1))]
		
		# Add splits to list
		W += list(P_w[(P_w.shape[0]-2)::-1])
		m_b = W[-1]
		b += 1
	
	# Last batch
	m_w = ceil((m - W[-1])/args.min_length) + 1
	X = np.zeros((m - W[-1], n), dtype=np.float32)
	L = np.zeros((m - W[-1], args.max_length), dtype=np.float32)
	E = np.zeros((m - W[-1], args.max_length), dtype=np.float32)
	C = np.full((m - W[-1], m_w), np.inf, dtype=np.float32)
	I = np.full((m - W[-1], m_w), -1, dtype=np.int32)
	P = np.zeros(m_w, dtype=np.int32)

	# Load batch
	split_cy.extractG(G, X, m_b, args.threads)

	# Estimate matrices and compute optimal paths
	split_cy.estimateL(X, L, args.threshold, args.threads)
	split_cy.estimateE(L, E)
	optK = split_cy.estimateC(E, C, I, args.min_length, args.threads)
	split_cy.constructP(I, P, m_b, optK)
	P = P[P <= (m_b + args.batch)]
	P = P[:(P.shape[0]-(np.sum(P == 0)-1))]
	
	# Add splits to list
	W += list(P[(P.shape[0]-2)::-1])

	# Save optimal window splits to file
	np.savetxt(f"{args.out}.opt.windows", np.array(W, dtype=int), fmt="%i")
	print(f"Saved optimal window indices in {args.out}.opt.windows")

	# Print elapsed time for parsing and total computation
	t_min = int(t_par//60)
	t_sec = int(t_par - t_min*60)
	print(f"Total parsing time: {t_min}m{t_sec}s")
	t_tot = time()-start
	t_min = int(t_tot//60)
	t_sec = int(t_tot - t_min*60)
	print(f"Total elapsed time: {t_min}m{t_sec}s")



##### Main exception #####
assert __name__ != "__main__", "Please use the 'hapla split' command!"
