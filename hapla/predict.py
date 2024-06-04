"""
hapla.
Haplotype clustering using pre-estimated cluster medians.
"""

__author__ = "Jonas Meisner"

# Libraries
import os
import re
from time import time

##### hapla predict #####
def main(args):
	print("-----------------------------------")
	print("hapla by Jonas Meisner (v0.9)")
	print(f"hapla predict using {args.threads} thread(s)")
	print("-----------------------------------\n")
	
	# Check input
	assert args.vcf is not None, \
		"Please provide phased genotype file (--bcf or --vcf)!"
	assert args.medians is not None, \
		"Please provide pre-estimated haplotype cluster medians (--medians)!"
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
	from hapla import memory_cy
	from hapla import shared_cy

	# Extract sample list
	print("\rLoading VCF/BCF file...", end="")
	v_file = VCF(args.vcf, threads=args.threads)
	m = 0
	n = 2*len(v_file.samples)
	B = ceil(n/4)
	if args.plink:
		v_list = []
		s_list = np.array(v_file.samples).reshape(-1, 1)

	# Check number of sites and allocate memory
	for variant in v_file:
		if args.plink:
			if m == 0:
				chrom = re.findall(r'\d+', variant.CHROM)[-1]
			v_list.append(variant.POS)
		m += 1
	if args.memory:
		G = np.zeros((m, B), dtype=np.uint8)
	else:
		G = np.zeros((m, n), dtype=np.uint8)

	# Read variants into matrix
	v_file = VCF(args.vcf, threads=args.threads)
	j = 0
	for variant in v_file:
		V = variant.genotype.array()
		if args.memory:
			memory_cy.predBit(G, V, j, n//2)
		else:
			reader_cy.predVar(G, V, j, n//2)
		j += 1
	del V
	print(f"\rLoaded phased genotype data: {n} haplotypes and {m} SNPs.")

	# Haplotype cluster medians
	M_npz = np.load(args.medians)
	w_vec = M_npz["W"]
	assert m == w_vec[-1], "SNP set does not match between files!"
	W = w_vec.shape[0] - 1
	print(f"Clustering {W} windows.")

	# Extract window information for PLINK output
	if args.plink:
		v_vec = np.array(v_list, dtype=np.int32)
		s_vec = v_vec[w_vec[:-1]].copy()
		b_vec = np.zeros(W, dtype=np.int32)
		del v_list, v_vec

	# Containers
	k_vec = np.zeros(W, dtype=np.uint8) # Number of clusters in windows
	Z = np.zeros((W, n), dtype=np.uint8) # Haplotype cluster alleles

	# Clustering
	for w in np.arange(W):
		print(f"\rWindow {w+1}/{W}", end="")

		# Load haplotype window
		M = M_npz[f"W{w}"]
		K = M.shape[0] # Number of clusters to evaluate
		X = np.zeros((n, M.shape[1]), dtype=np.uint8)
		if args.memory:
			memory_cy.predictBit(G, X, w_vec[w])
		else:
			reader_cy.predictHap(G, X, w_vec[w])
		if args.plink:
			b_vec[w] = M.shape[1]
		
		# Cluster assignment
		shared_cy.predictCluster(X, M, Z, K, w, args.threads)
		k_vec[w] = K
	del G, X, M, M_npz, w_vec
	print(".\n")

	# Save output
	np.save(f"{args.out}.z", Z)
	print(f"Saved predicted haplotype cluster alleles as {args.out}.z.npy")
	if args.plink:
		print("\rGenerating binary PLINK output.", end="")
		B = ceil(n/8)
		K_tot = np.sum(k_vec, dtype=int)
		P_mat = np.zeros((K_tot, 3), dtype=np.int32)
		Z_bin = np.zeros((K_tot, B), dtype=np.uint8)
		reader_cy.convertPlink(Z, Z_bin, P_mat, k_vec, b_vec)
		
		# Save .bed file including magic numbers
		with open(f"{args.out}.bed", "w") as bfile:
			np.array([108, 27, 1], dtype=np.uint8).tofile(bfile)
			Z_bin.tofile(bfile)
		del b_vec, Z_bin, Z

		# Save .bim file
		tmp = np.array([f"{chrom}_W{w}_K{k}_B{l}" for w,k,l in P_mat])
		bim = np.hstack((
			np.array([chrom]).repeat(K_tot).reshape(-1, 1), \
			tmp.reshape(-1, 1), np.zeros((K_tot, 1), dtype=np.int32), \
			s_vec.repeat(k_vec).reshape(-1, 1), \
			np.array(["K"]).repeat(K_tot).reshape(-1, 1), \
			np.zeros((K_tot, 1), dtype=np.int32)
		))
		np.savetxt(f"{args.out}.bim", bim, delimiter="\t", fmt="%s")
		del k_vec, s_vec, bim, tmp, P_mat
		
		# Save .fam file
		if args.duplicate_fid:
			s_list = s_list.repeat(2, axis=1)
		else:
			s_list = np.hstack((np.zeros((n//2, 1), dtype=np.uint8), s_list))
		fam = np.hstack((s_list, np.zeros((n//2, 3), dtype=np.uint8), \
			np.full((n//2, 1), -9, dtype=np.int8)))
		np.savetxt(f"{args.out}.fam", fam, delimiter="\t", fmt="%s")
		print("\rSaved haplotype cluster alleles in binary PLINK format:\n" + \
			f"- {args.out}.bed\n" + \
			f"- {args.out}.bim\n" + \
			f"- {args.out}.fam")
		del fam, s_list

	# Print elapsed time for computation
	t_tot = time()-start
	t_min = int(t_tot//60)
	t_sec = int(t_tot - t_min*60)
	print(f"Total elapsed time: {t_min}m{t_sec}s")



##### Main exception #####
assert __name__ != "__main__", "Please use the 'hapla predict' command!"
