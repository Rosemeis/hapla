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
	print("hapla by Jonas Meisner (v0.12)")
	print(f"hapla predict using {args.threads} thread(s)")
	print("-----------------------------------\n")
	
	# Check input
	assert args.vcf is not None, \
		"Please provide phased genotype file (--bcf or --vcf)!"
	assert os.path.isfile(f"{args.vcf}"), "VCF/BCF file doesn't exist!"
	assert args.ref is not None, \
		"Please provide pre-estimated reference haplotype cluster medians (--ref)!"
	assert os.path.isfile(f"{args.ref}.bcm"), "bcm file doesn't exist!"
	assert os.path.isfile(f"{args.ref}.win"), "win file doesn't exist!"
	assert os.path.isfile(f"{args.ref}.wix"), "wix file doesn't exist!"
	assert os.path.isfile(f"{args.ref}.hcc"), "hcc file doesn't exist!"
	assert args.threads > 0, "Please select a valid number of threads!"
	start = time()

	# Control threads of external numerical libraries
	os.environ["MKL_NUM_THREADS"] = str(args.threads)
	os.environ["MKL_MAX_THREADS"] = str(args.threads)
	os.environ["OMP_NUM_THREADS"] = str(args.threads)
	os.environ["OMP_MAX_THREADS"] = str(args.threads)
	os.environ["NUMEXPR_NUM_THREADS"] = str(args.threads)
	os.environ["NUMEXPR_MAX_THREADS"] = str(args.threads)
	os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)
	os.environ["OPENBLAS_MAX_THREADS"] = str(args.threads)

	# Import numerical libraries and cython functions
	import numpy as np
	from cyvcf2 import VCF
	from math import ceil
	from hapla import reader_cy
	from hapla import memory_cy
	from hapla import shared_cy

	# Initiate VCF and extract sample list
	print("\rLoading VCF/BCF file...", end="")
	v_file = VCF(args.vcf, threads=args.threads)
	v_list = []
	s_list = np.array(v_file.samples).reshape(-1,1)
	m = 0
	n = 2*len(v_file.samples)
	B = ceil(n/4)

	# Check number of sites and allocate memory
	for variant in v_file:
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

	# Load window information from reference
	w_mat = np.loadtxt(f"{args.ref}.win", dtype=np.str_)
	s_vec = w_mat[:,1].astype(np.int32)
	b_vec = w_mat[:,4].astype(np.int32)
	k_vec = w_mat[:,5].astype(np.uint8)
	W = k_vec.shape[0]

	# Haplotype cluster medians
	with open(f"{args.ref}.bcm", "rb") as f:
		# Check magic numbers
		m_vec = np.fromfile(f, dtype=np.uint8, count=3)
		assert np.allclose(m_vec, np.array([7, 9, 13], dtype=np.uint8)), \
			"Magic number doesn't match file format!"
		M_arr = np.fromfile(f, dtype=np.uint8)
		
	# Load window setup files
	w_vec = np.loadtxt(f"{args.ref}.wix", dtype=np.int32)
	N_arr = np.loadtxt(f"{args.ref}.hcc", dtype=np.int32)
	v_vec = np.array(v_list, dtype=np.int32)
	assert np.allclose(v_vec[w_vec], s_vec), "SNP set doesn't match!"
	del v_list, v_vec

	# Containers
	Z = np.zeros((W, n), dtype=np.uint8) # Haplotype cluster alleles

	# Clustering
	B = 0
	K = 0
	print(f"Clustering {W} windows.")
	for w in np.arange(W):
		print(f"\rWindow {w+1}/{W}", end="")
		b_win = int(b_vec[w])
		k_win = int(k_vec[w])

		# Load haplotype window
		M_mat = M_arr[B:(B + k_win*b_win)]
		M_mat.shape = (k_win, b_win)
		n_vec = N_arr[K:(K + k_win)]
		X = np.zeros((n, M_mat.shape[1]), dtype=np.uint8)
		if args.memory:
			memory_cy.predictBit(G, X, w_vec[w])
		else:
			reader_cy.predictHap(G, X, w_vec[w])
		
		# Cluster assignment
		shared_cy.predictCluster(X, M_mat, Z, n_vec, k_vec[w], w, args.threads)

		# Update counter
		B += k_win*b_win
		K += k_win
	del G, X, M_mat, M_arr, N_arr, n_vec, w_vec
	print(".\n")

	# Save hapla output
	with open(f"{args.out}.bca", "wb") as f:
		np.array([7, 9, 13], dtype=np.uint8).tofile(f) # Add magic numbers
		Z.tofile(f) # Save haplotype cluster assignments to binary file
	np.savetxt(f"{args.out}.ids", s_list, fmt="%s")
	np.savetxt(f"{args.out}.win", w_mat, delimiter="\t", fmt="%s")
	print("\rSaved haplotype clusters in binary hapla format:\n" + \
		f"- {args.out}.bca\n" + \
		f"- {args.out}.ids\n" + \
		f"- {args.out}.win\n")

	# Save haplotype cluster assignments in binary PLINK format
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
			np.array([chrom]).repeat(K_tot).reshape(-1,1), \
			tmp.reshape(-1,1), \
			np.zeros((K_tot, 1), dtype=np.int32), \
			s_vec.repeat(k_vec).reshape(-1,1), \
			np.array(["K"]).repeat(K_tot).reshape(-1,1), \
			np.zeros((K_tot, 1), dtype=np.int32)
		))
		np.savetxt(f"{args.out}.bim", bim, delimiter="\t", fmt="%s")
		del k_vec, s_vec, bim, tmp, P_mat
		
		# Save .fam file
		if args.duplicate_fid:
			s_list = s_list.repeat(2, axis=1)
		else:
			s_list = np.hstack((np.zeros((n//2, 1), dtype=np.uint8), s_list))
		fam = np.hstack((
			s_list, \
			np.zeros((n//2, 3), dtype=np.uint8), \
			np.full((n//2, 1), -9, dtype=np.int8)
		))
		np.savetxt(f"{args.out}.fam", fam, delimiter="\t", fmt="%s")
		print("\rSaved haplotype cluster alleles in binary PLINK format:\n" + \
			f"- {args.out}.bed\n" + \
			f"- {args.out}.bim\n" + \
			f"- {args.out}.fam\n")
		del fam, s_list

	# Print elapsed time for computation
	t_tot = time()-start
	t_min = int(t_tot//60)
	t_sec = int(t_tot - t_min*60)
	print(f"Total elapsed time: {t_min}m{t_sec}s")



##### Main exception #####
assert __name__ != "__main__", "Please use the 'hapla predict' command!"
