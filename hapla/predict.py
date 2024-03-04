"""
hapla.
Haplotype clustering using pre-estimated cluster medians.
"""

__author__ = "Jonas Meisner"

# Libraries
import os
from time import time

##### hapla predict #####
def main(args):
	print("-----------------------------------")
	print("hapla by Jonas Meisner (v0.5)")
	print(f"hapla predict using {args.threads} thread(s)")
	print("-----------------------------------\n")
	
	# Check input
	assert args.vcf is not None, \
		"Please provide phased genotype file (--bcf or --vcf)!"
	assert args.medians is not None, \
		"Please provide pre-estimated haplotype cluster medians (--medians)!"
	if args.plink:
		assert args.counts is not None, \
			"Please provide pre-estimated haplotype cluster counts (--counts)!"
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
	from hapla import shared_cy

	# Load data into 2-bit matrix
	print("\rLoading VCF/BCF file...", end="")
	v_file = VCF(args.vcf, threads=args.threads)
	m = 0
	n = 2*len(v_file.samples)
	B = ceil(n/4)

	# Check number of sites and allocate memory
	for variant in v_file:
		m += 1
	G = np.zeros((m, B), dtype=np.uint8)

	# Read variants into matrix
	v_file = VCF(args.vcf, threads=args.threads)
	j = 0
	for variant in v_file:
		V = variant.genotype.array()
		reader_cy.readPred(G, V, j, n//2)
		j += 1
	del V
	print(f"\rLoaded phased genotype data: {n} haplotypes and {m} SNPs.")

	# Haplotype cluster medians
	M_npz = np.load(args.medians)
	m_ref, win, W, overlap = list(M_npz["I"])
	assert m == m_ref, "SNP set does not match between files!"

	# Setup windows
	if overlap > 0:
		W_vec = [w*(win//(overlap + 1)) for w in range(W)]
		print(f"Clustering {W} overlapping windows of {win} SNPs.")
	else:
		W_vec = [w*win for w in range(W)]
		print(f"Clustering {W} non-overlapping windows of {win} SNPs.")
	W_vec = np.array(W_vec, dtype=int)

	# Haplotype cluster counts
	if args.plink:
		N_npz = np.load(args.counts)
		R_vec = np.zeros(W, dtype=np.uint8) # Rarest clusters in windows

	# Containers
	K_vec = np.zeros(W, dtype=np.uint8) # Number of clusters in windows
	H = np.zeros((win, n), dtype=np.uint8) # Haplotypes
	X = np.zeros((n, win), dtype=np.uint8) # Haplotypes transposed
	Z_mat = np.zeros((W, n), dtype=np.uint8) # Haplotype cluster alleles

	# Clustering
	for w in np.arange(W):
		# Load haplotype window
		M = M_npz[f"W{w}"]
		K = M.shape[0] # Number of clusters to evaluate
		if w == (W-1): # Last window
			H = np.zeros((M.shape[1], n), dtype=np.uint8)
			X = np.zeros((n, M.shape[1]), dtype=np.uint8)
		reader_cy.predictBit(G, H, W_vec[w])
		
		# Transposed in contiguous memory
		np.copyto(X, H.T, casting="no")
		
		# Cluster assignment
		shared_cy.predictCluster(X, M, Z_mat, K, w, args.threads)
		K_vec[w] = K
		if args.plink:
			R_vec[w] = np.argmin(N_npz[f"W{w}"])
	del G

	# Save output
	np.save(f"{args.out}.z", Z_mat)
	print(f"Saved predicted haplotype cluster alleles as {args.out}.z.npy")
	if args.plink:
		print("\rGenerating binary PLINK output.", end="")
		import re
		v_file = VCF(args.vcf, threads=args.threads)
		s_list = np.array(v_file.samples).reshape(-1,1)
		for variant in v_file: # Extract chromosome name from first entry
			chrom = re.findall(r'\d+', variant.CHROM)[-1]
			break
		del v_file
		B = ceil(n/8)
		K_tot = np.sum(K_vec-1, dtype=int)
		P_mat = np.zeros((K_tot, 2), dtype=np.int32)
		Z_vec = np.zeros(n//2, dtype=np.uint8)
		Z_bin = np.zeros((K_tot, B), dtype=np.uint8)
		reader_cy.convertPlink(Z_mat, Z_bin, P_mat, Z_vec, R_vec, K_vec)
		
		# Save .bed file including magic numbers
		with open(f"{args.out}.bed", "w") as bfile:
			np.array([108, 27, 1], dtype=np.uint8).tofile(bfile)
			Z_bin.tofile(bfile)
		del K_vec, Z_bin, Z_mat, Z_vec

		# Save .bim file
		tmp = np.array([f"{chrom}_B{args.win}_W{w}_K{k}" for w,k in P_mat])
		bim = np.hstack((np.array([chrom]).repeat(K_tot).reshape(-1,1), \
			tmp.reshape(-1,1), np.zeros((K_tot, 1), dtype=np.uint8), \
			np.arange(1, K_tot+1).reshape(-1,1), \
			np.array(["K"]).repeat(K_tot).reshape(-1,1), \
			np.zeros((K_tot, 1), dtype=np.uint8)))
		np.savetxt(f"{args.out}.bim", bim, delimiter="\t", fmt="%s")
		del bim, tmp, P_mat
		
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

	# Print elapsed time for estimation
	t_tot = time()-start
	t_min = int(t_tot//60)
	t_sec = int(t_tot - t_min*60)
	print(f"Total elapsed time: {t_min}m{t_sec}s")



##### Main exception #####
assert __name__ != "__main__", "Please use the 'hapla predict' command!"
