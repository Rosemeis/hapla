"""
hapla.
Haplotype clustering using K-Medians with delayed cluster creation.
"""

__author__ = "Jonas Meisner"

# Libraries
import os
import re
from time import time

##### hapla cluster #####
def main(args):
	print("-----------------------------------")
	print("hapla by Jonas Meisner (v0.12)")
	print(f"hapla cluster using {args.threads} thread(s)")
	print("-----------------------------------\n")
	
	# Check input
	assert args.vcf is not None, \
		"No phased genotype file (--bcf or --vcf)!"
	assert os.path.isfile(f"{args.vcf}"), "VCF/BCF file doesn't exist!"
	assert args.threads > 0, "Please select a valid number of threads!"
	assert args.min_freq > 0.0, "Invalid haplotype cluster frequency!"
	assert args.max_iterations > 0, "Please select a valid number of iterations!"
	assert (args.lmbda > 0.0) and (args.lmbda < 1.0), \
		"Please select a valid lambda value!"
	assert (args.max_clusters > 1) and (args.max_clusters <= 256), \
		"Max allowed clusters exceeded!"
	if args.fixed is not None:
		assert args.fixed > 0, "Invalid window size!"
		if args.step is not None:
			assert (args.step <= args.fixed) and (args.step > 0), \
				"Invalid step size for sliding window chosen!"
			if args.fixed == 1:
				args.step = None
	else:
		assert args.windows is not None, "No window option (--fixed or --windows)!"
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
	from hapla import cluster_cy

	# Initiate VCF and extract sample list
	print("\rLoading VCF/BCF file...", end="")
	v_file = VCF(args.vcf, threads=args.threads)
	v_list = []
	s_list = np.array(v_file.samples).reshape(-1,1)
	m = 0
	n = 2*len(v_file.samples)
	B = ceil(n/8)

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
			memory_cy.readBit(G, V, j, n//2)
		else:
			reader_cy.readVar(G, V, j, n//2)
		j += 1
	del V
	t_par = time()-start
	print(f"\rLoaded phased genotype data: {n} haplotypes and {m} SNPs.")

	# Check haplotype cluster frequency
	N_mac = ceil(n*args.min_freq)
	assert N_mac > 2, "Frequency threshold too low for sample size (--min-freq)!"

	# Setup windows
	if args.fixed is not None:
		if args.step is not None:
			W = ceil((m - args.fixed)/args.step)
			w_vec = [w*args.step for w in range(W)]
			print(f"Clustering {W} overlapping windows of {args.fixed} SNPs " + \
		 			f"(step-size {args.step}).")
		else:
			W = m//args.fixed
			w_vec = [w*args.fixed for w in range(W)]
			print(f"Clustering {W} non-overlapping windows of {args.fixed} SNPs.")
		w_vec.append(m)
		w_vec = np.array(w_vec, dtype=np.int32)
	else:
		w_vec = np.loadtxt(args.windows, dtype=np.int32)
		assert w_vec[-1] <= m, "Genotype and window files don't match!"
		W = w_vec.shape[0]
		w_vec = np.insert(w_vec, W, m)
		print(f"Clustering {W} windows with provided SNP lengths.")

	# Extract window information
	v_vec = np.array(v_list, dtype=np.int32)
	s_vec = v_vec[w_vec[:-1]].copy()
	if args.fixed is not None:
		e_vec = v_vec[w_vec[:-1]+args.fixed-1].copy()
		e_vec[-1] = v_vec[-1]
		b_vec = np.full(W, args.fixed, dtype=np.int32)
		b_vec[-1] = w_vec[-1] - w_vec[-2]
	else:
		e_vec = v_vec[w_vec[1:]-1]
		b_vec = w_vec[1:] - w_vec[:-1]
	del v_list, v_vec

	# Containers
	z_vec = np.zeros(n, dtype=np.uint8) # Window-based cluster assignments 
	k_vec = np.zeros(W, dtype=np.uint8) # Number of clusters in windows
	c_vec = np.zeros(n, dtype=np.int32) # Cost vector
	u_vec = np.zeros(n, dtype=np.int32) # Count of unique haplotypes
	d_vec = np.zeros(n, dtype=np.int32) # Divergence vector (suffix array)
	p_vec = np.arange(n, dtype=np.int32) # Prefix vector (suffix array)
	n_vec = np.zeros(args.max_clusters, dtype=np.int32) # Size vector
	z_tmp = np.zeros_like(z_vec) # Help vector (clustering)
	a_tmp = np.zeros_like(p_vec) # Help vector (suffix array)
	b_tmp = np.zeros_like(p_vec) # Help vector (suffix array)
	d_tmp = np.zeros_like(d_vec) # Help vector (suffix array)
	e_tmp = np.zeros_like(d_vec) # Help vector (suffix array)
	Z = np.zeros((W, n), dtype=np.uint8) # Chromosome-based cluster assignments
	if args.fixed is not None: # Window length-based
		if args.memory:
			H = np.zeros((args.fixed, n), dtype=np.uint8) # Haplotypes transposed
		X = np.zeros((n, args.fixed), dtype=np.uint8) # Haplotypes
		M = np.zeros((args.max_clusters, args.fixed), dtype=np.uint8) # Medians
		C = np.zeros((args.max_clusters, args.fixed), dtype=np.float32) # Means

	# Thread-local containers
	I_thr = np.zeros((args.threads, 2), dtype=np.int32)
	N_thr = np.zeros((args.threads, args.max_clusters), dtype=np.int32)
	if args.fixed is not None:
		C_thr = np.zeros((args.threads, args.max_clusters, args.fixed), \
			dtype=np.float32)

	# Optional containers
	if args.medians:
		N_arr = np.array([], dtype=np.int32)
		with open(f"{args.out}.bcm", "wb") as f:
			np.array([7, 9, 13], dtype=np.uint8).tofile(f)
		np.savetxt(f"{args.out}.wix", w_vec[:-1], fmt="%i")

	# Clustering using PDC-DP-Medians
	for w in np.arange(W):
		s = w_vec[w]
		print(f"\rWindow {w+1}/{W}", end="")

		# Prepare containers if window indices provided
		if args.fixed is None:
			if args.memory:
				H = np.zeros((w_vec[w+1]-s, n), dtype=np.uint8)
			X = np.zeros((n, w_vec[w+1]-s), dtype=np.uint8)
			M = np.zeros((args.max_clusters, X.shape[1]), dtype=np.uint8)
			C = np.zeros((args.max_clusters, X.shape[1]), dtype=np.float32)
			C_thr = np.zeros((args.threads, args.max_clusters, X.shape[1]), \
				dtype=np.float32)

		# Prepare last window
		if w == (W-1):
			if args.memory:
				H = np.zeros((m-s, n), dtype=np.uint8)
			X = np.zeros((n, m-s), dtype=np.uint8)
			M = np.zeros((args.max_clusters, X.shape[1]), dtype=np.uint8)
			C = np.zeros((args.max_clusters, X.shape[1]), dtype=np.float32)
			C_thr = np.zeros((args.threads, args.max_clusters, X.shape[1]), \
				dtype=np.float32)
		c_lim = args.lmbda*X.shape[1]
		
		# Load haplotype window
		if args.memory:
			memory_cy.convertBit(G, H, C, p_vec, d_vec, a_tmp, b_tmp, d_tmp, e_tmp, s)
			U = memory_cy.uniqueBit(H, X, p_vec, d_vec, u_vec)
		else:
			reader_cy.convertHap(G, C, p_vec, d_vec, a_tmp, b_tmp, d_tmp, e_tmp, s)
			U = reader_cy.uniqueHap(G, X, p_vec, d_vec, u_vec, s)
		T = min(U, args.threads)
		reader_cy.intervalThr(I_thr, U, U//T)

		# Compute mean and initialize first median
		K = 1
		n_vec[0] = n
		cluster_cy.marginalMedians(M, C, n_vec, K)

		# Perform PDC-DP-Medians
		for it in np.arange(args.max_iterations):
			cluster_cy.clusterAssignment(X, M, z_vec, c_vec, n_vec, u_vec, \
				C_thr, N_thr, I_thr, K, T)
			cluster_cy.updateArrays(C_thr, C, N_thr, n_vec, K, T)
			K += cluster_cy.checkCluster(X, M, C, z_vec, c_vec, n_vec, u_vec, c_lim, K)

			# Check for convergence
			if it > 0:
				if cluster_cy.countDist(z_vec, z_tmp) == 0:
					if K > 1:
						break
					else: # Make sure two haplotype clusters are generated
						print(", No diversity (K=1)! Adding extra cluster.")
						cluster_cy.genCluster(X, M, C, z_vec, c_vec, n_vec, u_vec, K)
						K += 1
			else:
				memoryview(z_tmp)[:] = z_vec # memcpy
			
			# Count sizes and construct marginal medians
			cluster_cy.marginalMedians(M, C, n_vec, K)

		# Iterative re-clustering of haplotypes
		if K > 2:
			# Ensure correct medians
			cluster_cy.marginalMedians(M, C, n_vec, K)

			# Remove singletons in one go
			n_vec[n_vec == 1] = 0
			cluster_cy.clusterAssignment(X, M, z_vec, c_vec, n_vec, u_vec, \
				C_thr, N_thr, I_thr, K, T)
			cluster_cy.updateArrays(C_thr, C, N_thr, n_vec, K, T)
			K_tmp = np.sum(n_vec > 0)

			# Remove small clusters iterativly
			while K_tmp > 2:
				# Re-assign haplotypes
				cluster_cy.marginalMedians(M, C, n_vec, K)
				cluster_cy.clusterAssignment(X, M, z_vec, c_vec, n_vec, u_vec, \
					C_thr, N_thr, I_thr, K, T)
				cluster_cy.updateArrays(C_thr, C, N_thr, n_vec, K, T)

				# Find smallest cluster
				N_min = cluster_cy.findZero(n_vec, n, N_mac, K)
				if N_min >= N_mac: # Ensure convergence
					if cluster_cy.countDist(z_vec, z_tmp) == 0:
						break
				else:
					K_tmp -= 1
				memoryview(z_tmp)[:] = z_vec # memcpy

			# Re-cluster K = 2 non-break case
			if (K_tmp == 2) and (N_min < N_mac):
				cluster_cy.clusterAssignment(X, M, z_vec, c_vec, n_vec, u_vec, \
					C_thr, N_thr, I_thr, K, T)
				np.sum(N_thr, axis=0, out=n_vec)

		# Fix cluster median and cluster assignment order
		cluster_cy.medianFix(M, z_vec, n_vec, K, U)
		cluster_cy.assignFix(Z, z_vec, p_vec, d_vec, w)
		K = np.sum(n_vec > 0, dtype=int)
		k_vec[w] = K

		# Generate optional saves (medians)
		if args.medians:
			N_arr = np.append(N_arr, n_vec[:K])
			with open(f"{args.out}.bcm", "ab") as f:
				M[:K].tofile(f)

		# Reset arrays
		n_vec.fill(0)
		cluster_cy.resetArrays(C_thr, N_thr, c_vec, p_vec, d_vec, u_vec, T)
	
	# Release memory
	del G, X, C_thr, N_thr, w_vec, z_vec, c_vec, z_tmp, p_vec, d_vec, n_vec, u_vec, \
		a_tmp, b_tmp, d_tmp, e_tmp
	if args.memory:
		del H
	print(".\n")
	
	# Create window information array
	w_mat = np.hstack((
		np.array([chrom]).repeat(W).reshape(-1,1), \
		s_vec.reshape(-1,1), e_vec.reshape(-1,1), (e_vec - s_vec).reshape(-1,1), \
		b_vec.reshape(-1,1), k_vec.reshape(-1,1)
	))

	# Save hapla output
	with open(f"{args.out}.bca", "wb") as f:
		np.array([7, 9, 13], dtype=np.uint8).tofile(f) # Add magic numbers
		Z.tofile(f) # Save haplotype cluster assignments to binary file format
	np.savetxt(f"{args.out}.ids", s_list, fmt="%s")
	np.savetxt(f"{args.out}.win", w_mat, delimiter="\t", fmt="%s")
	print("\rSaved haplotype clusters in binary hapla format:\n" + \
		f"- {args.out}.bca\n" + \
		f"- {args.out}.ids\n" + \
		f"- {args.out}.win\n")
	if args.medians: # Save haplotype cluster medians to binary file format
		np.savetxt(f"{args.out}.hcc", N_arr, fmt="%i")
		print(f"Saved haplotype cluster medians in binary format:\n" + \
			f"- {args.out}.bcm\n" + \
			f"- {args.out}.wix\n" + \
			f"- {args.out}.hcc\n")
		del N_arr
	del e_vec, w_mat

	# Save haplotype cluster assignments in binary PLINK format
	if args.plink:
		print("\rGenerating binary PLINK output.", end="")
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
		print("\rSaved haplotype clusters in binary PLINK format:\n" + \
			f"- {args.out}.bed\n" + \
			f"- {args.out}.bim\n" + \
			f"- {args.out}.fam\n")
		del fam, s_list

	# Print elapsed time for parsing and total computation
	t_min = int(t_par//60)
	t_sec = int(t_par - t_min*60)
	print(f"Total parsing time: {t_min}m{t_sec}s")
	t_tot = time()-start
	t_min = int(t_tot//60)
	t_sec = int(t_tot - t_min*60)
	print(f"Total elapsed time: {t_min}m{t_sec}s")

	

##### Main exception #####
assert __name__ != "__main__", "Please use the 'hapla cluster' command!"
