"""
hapla.
Haplotype clustering using PDC-DP-Medians.
"""

__author__ = "Jonas Meisner"

# Libraries
import os
from datetime import datetime
from time import time

VERSION = "0.32.1"

##### hapla cluster #####
def main(args, deaf):
	print("-----------------------------------")
	print(f"hapla by Jonas Meisner (v{VERSION})")
	print(f"hapla cluster using {args.threads} thread(s)")
	print("-----------------------------------\n")

	# Check input
	assert args.vcf is not None, "No phased genotype file (--bcf or --vcf)!"
	assert os.path.isfile(f"{args.vcf}"), "VCF/BCF file doesn't exist!"
	assert os.path.isfile(f"{args.vcf}.csi") or os.path.isfile(f"{args.vcf}.tbi"), "VCF/BCF index doesn't exist!"
	assert args.threads > 0, "Please select a valid number of threads!"
	assert args.min_freq > 0.0, "Invalid haplotype cluster frequency!"
	if args.min_mac is not None:
		assert args.min_mac > 1, "Please select a valid MAC threshold!"
	assert args.max_iterations > 0, "Please select a valid number of iterations!"
	assert (args.lmbda > 0.0) and (args.lmbda < 1.0), "Please select a valid lambda value!"
	assert (args.max_clusters > 1) and (args.max_clusters <= 256), "Max allowed clusters exceeded!"
	if args.size is not None:
		assert args.size > 0, "Invalid window size!"
		if args.step is not None:
			if args.size == 1:
				args.step = None
			else:
				assert (args.step <= args.size) and (args.step > 0), "Invalid step size for sliding window chosen!"
	else:
		assert args.windows is not None, "No window option (--size or --windows)!"
	start = time()

	# Create log-file of used arguments
	full = vars(args)
	mand = ["lmbda"]
	if args.min_mac is None:
		mand.append("min_freq")
	with open(f"{args.out}.log", "w") as log:
		log.write(f"hapla v{VERSION}\n")
		log.write("hapla cluster\n")
		log.write(f"Time: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
		log.write(f"Directory: {os.getcwd()}\n")
		log.write("Options:\n")
		for key in full:
			if full[key] != deaf[key]:
				if type(full[key]) is bool:
					log.write(f"\t--{key}\n")
				else:
					log.write(f"\t--{key} {full[key]}\n")
			elif key in mand:
				log.write(f"\t--{key} {full[key]}\n")
	del full, deaf, mand
	
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

	# Initiate VCF and extract parameters
	print("\rLoading VCF/BCF file...", end="")
	v_file = VCF(args.vcf, threads=min(args.threads, 4))
	s_list = np.array(v_file.samples).reshape(-1, 1)
	N = 2*s_list.shape[0]
	M = v_file.num_records
	B = ceil(N/8)

	# Set haplotype cluster size threshold
	if args.min_mac is not None:
		N_mac = np.uint32(args.min_mac)
	else:
		N_mac = np.uint32(ceil(N*args.min_freq))

	# Allocate arrays
	if args.memory:
		G = np.zeros((M, B), dtype=np.uint8)
	else:
		G = np.zeros((M, N), dtype=np.uint8)
	v_vec = np.zeros(M, dtype=np.uint32)

	# Read variants into matrix
	for j, variant in enumerate(v_file):
		V = variant.genotype.array()
		if args.memory:
			memory_cy.readBit(G[j], V, N//2)
		else:
			reader_cy.readVar(G[j], V, N//2)
		v_vec[j] = variant.POS
	chrom = variant.CHROM # Extract chromosome information
	del V, v_file
	t_par = time() - start
	print(f"\rLoaded phased genotype data: {N} haplotypes and {M} SNPs.")

	# Set up windows
	if args.size is not None:
		if args.step is not None:
			W = ceil((M - args.size)/args.step)
			w_vec = [w*args.step for w in range(W)]
			print(f"Clustering {W} overlapping windows of {args.size} SNPs " + \
		 			f"(step-size {args.step}).")
		else:
			W = M//args.size
			w_vec = [w*args.size for w in range(W)]
			print(f"Clustering {W} non-overlapping windows of {args.size} SNPs.")
		w_vec.append(M)
		w_vec = np.array(w_vec, dtype=np.uint32)
	else:
		w_vec = np.genfromtxt(args.windows, dtype=np.uint32)
		assert w_vec[-1] <= M, "Genotype and window files don't match!"
		if w_vec[-1] != M:
			w_vec = np.insert(w_vec, W, M)
		W = w_vec.shape[0]
		print(f"Clustering {W} windows with provided SNP lengths.")

	# Containers
	Z = np.zeros((W, N), dtype=np.uint8) # Chromosome-based cluster assignments
	z_vec = np.zeros(N, dtype=np.uint8) # Window-based cluster assignments 
	k_vec = np.zeros(W, dtype=np.uint32) # Number of clusters in windows
	c_vec = np.zeros(N, dtype=np.uint32) # Cost vector
	u_vec = np.zeros(N, dtype=np.uint32) # Count of unique haplotypes
	d_vec = np.zeros(N, dtype=np.uint32) # Divergence vector (suffix array)
	n_vec = np.zeros(args.max_clusters, dtype=np.uint32) # Size vector
	p_vec = np.arange(N, dtype=np.uint32) # Prefix vector (suffix array)
	i_vec = np.arange(args.max_iterations) # Iteration vector
	z_tmp = np.zeros_like(z_vec) # Help vector (clustering)
	a_tmp = np.zeros_like(p_vec) # Help vector (suffix array)
	b_tmp = np.zeros_like(p_vec) # Help vector (suffix array)
	d_tmp = np.zeros_like(d_vec) # Help vector (suffix array)
	e_tmp = np.zeros_like(d_vec) # Help vector (suffix array)
	n_tmp = np.zeros_like(n_vec) # Help vector (size)
	if args.size is not None: # Window length-based
		if args.memory:
			H = np.zeros((args.size, N), dtype=np.uint8) # Haplotypes transposed
		X = np.zeros((N, args.size), dtype=np.uint8) # Haplotypes
		R = np.zeros((args.max_clusters, args.size), dtype=np.uint8) # Medians
		C = np.zeros((args.max_clusters, args.size), dtype=np.uint32) # Means
		c_lim = np.uint32(ceil(args.lmbda*float(X.shape[1]))) # SNP-based threshold

	# Optional containers
	if args.medians:
		L = np.zeros((args.max_clusters, args.max_clusters), dtype=np.float32) # Log-likelihoods
		with open(f"{args.out}.bcm", "wb") as f: # Medians file
			np.array([7, 9, 13], dtype=np.uint8).tofile(f)
		with open(f"{args.out}.blk", "wb") as f: # Log-likelihoods file
			np.array([7, 9, 13], dtype=np.uint8).tofile(f)
		np.savetxt(f"{args.out}.wix", w_vec[:-1], fmt="%i") # Window lengths

	# Clustering using PDC-DP-Medians
	for w in np.arange(W):
		S = w_vec[w]
		print(f"\rWindow {w + 1}/{W}", end="")

		# Prepare containers if window indices provided
		if args.size is None:
			if args.memory:
				H = np.zeros((w_vec[w + 1] - S, N), dtype=np.uint8)
			X = np.zeros((N, w_vec[w + 1] - S), dtype=np.uint8)
			R = np.zeros((args.max_clusters, X.shape[1]), dtype=np.uint8)
			C = np.zeros((args.max_clusters, X.shape[1]), dtype=np.uint32)
			c_lim = np.uint32(ceil(args.lmbda*float(X.shape[1])))

		# Prepare last window
		if w == (W-1):
			if args.memory:
				H = np.zeros((M - S, N), dtype=np.uint8)
			X = np.zeros((N, M - S), dtype=np.uint8)
			R = np.zeros((args.max_clusters, X.shape[1]), dtype=np.uint8)
			C = np.zeros((args.max_clusters, X.shape[1]), dtype=np.uint32)
			c_lim = np.uint32(ceil(args.lmbda*float(X.shape[1])))

		# Load haplotype window
		if args.memory:
			memory_cy.convertBit(G, H, C, p_vec, d_vec, a_tmp, b_tmp, d_tmp, e_tmp, S)
			U = memory_cy.uniqueBit(H, X, p_vec, d_vec, u_vec)
		else:
			reader_cy.convertHap(G, C, p_vec, d_vec, a_tmp, b_tmp, d_tmp, e_tmp, S)
			U = reader_cy.uniqueHap(G, X, p_vec, d_vec, u_vec, S)

		# Compute mean and initialize first median
		K = np.uint32(1)
		n_vec[0] = N
		cluster_cy.marginalMedians(R, C, n_vec, K)

		# Perform PDC-DP-Medians
		for it in i_vec:
			cluster_cy.assignClust(X, R, C, z_vec, c_vec, n_vec, n_tmp, u_vec, U, K)
			cluster_cy.updateN(n_vec, n_tmp, K)
			K += cluster_cy.checkClust(X, R, C, z_vec, c_vec, n_vec, u_vec, c_lim, U, K)

			# Check for convergence
			if it > 0:
				if cluster_cy.countDist(z_vec, z_tmp, U) == 0:
					if K > 1: # Converged
						break
					else: # Make sure two haplotype clusters are generated
						print(", No diversity (K=1)! Adding extra cluster.")
						cluster_cy.genClust(X, R, C, z_vec, c_vec, n_vec, u_vec, U, K)
						K += 1
			else:
				memoryview(z_tmp)[:] = memoryview(z_vec)

			# Count sizes and construct marginal medians
			cluster_cy.marginalMedians(R, C, n_vec, K)

		# Iterative re-clustering of haplotypes
		if K > 2:
			# Remove smallest clusters iterativly
			K_tmp = np.sum(n_vec > 0, dtype=np.uint32)
			while K_tmp > 2:
				# Re-assign haplotypes
				cluster_cy.marginalMedians(R, C, n_vec, K)
				cluster_cy.assignClust(X, R, C, z_vec, c_vec, n_vec, n_tmp, u_vec, U, K)
				cluster_cy.updateN(n_vec, n_tmp, K)

				# Find smallest cluster
				N_min = cluster_cy.findZero(n_vec, N, N_mac, K)
				if N_min >= N_mac: # Ensure convergence
					if cluster_cy.countDist(z_vec, z_tmp, U) == 0:
						break
				else:
					K_tmp -= 1 # Cluster removed
					memoryview(z_tmp)[:] = memoryview(z_vec)

			# Re-cluster K = 2 case
			if (K_tmp == 2) and (N_min < N_mac):
				cluster_cy.assignClust(X, R, C, z_vec, c_vec, n_vec, n_tmp, u_vec, U, K)
				cluster_cy.updateN(n_vec, n_tmp, K)

		# Fix cluster median and cluster assignment order
		cluster_cy.medianFix(R, C, z_vec, n_vec, K, U)
		cluster_cy.assignFix(Z, z_vec, p_vec, d_vec, w)
		K = np.sum(n_vec > 0, dtype=np.uint32)
		k_vec[w] = K

		# Generate optional saves (medians)
		if args.medians:
			cluster_cy.estimateLoglike(R, C, L, n_vec, K)
			with open(f"{args.out}.bcm", "ab") as f:
				R[:K].tofile(f)
			with open(f"{args.out}.blk", "ab") as f:
				L[:K,:K].tofile(f)

		# Reset arrays
		cluster_cy.resetArrays(c_vec, n_vec, p_vec, d_vec, u_vec)

	# Release memory
	del G, X, C, z_vec, c_vec, u_vec, d_vec, n_vec, p_vec, i_vec, z_tmp, a_tmp, b_tmp, d_tmp, e_tmp, n_tmp
	if args.memory:
		del H
	print(".\n")

	# Extract window information
	s_vec = v_vec[w_vec[:-1]].copy()
	if args.size is not None:
		e_vec = v_vec[w_vec[:-1] + args.size-1].copy()
		e_vec[-1] = v_vec[-1]
		b_vec = np.full(W, args.size, dtype=np.uint32)
		b_vec[-1] = w_vec[-1] - w_vec[-2]
	else:
		e_vec = v_vec[w_vec[1:] - 1]
		b_vec = w_vec[1:] - w_vec[:-1]
	del v_vec, w_vec

	# Create window information array
	w_mat = np.hstack((
		np.array([chrom]).repeat(W).reshape(-1, 1),
		s_vec.reshape(-1, 1), e_vec.reshape(-1, 1), 
		(e_vec - s_vec).reshape(-1, 1),
		b_vec.reshape(-1, 1), k_vec.reshape(-1, 1)
	))

	# Save hapla output and print info
	h_win = ["#CHROM", "START", "END", "LENGTH", "SIZE", "K"]
	with open(f"{args.out}.bca", "wb") as f:
		np.array([7, 9, 13], dtype=np.uint8).tofile(f) # Add magic numbers
		Z.tofile(f) # Save haplotype cluster assignments to binary file format
	np.savetxt(f"{args.out}.ids", s_list, fmt="%s")
	np.savetxt(f"{args.out}.win", w_mat, fmt="%s", delimiter="\t", comments="", header="\t".join(h_win))
	print("Saved haplotype clusters in binary format:\n" + \
		f"- {args.out}.bca\n" + \
		f"- {args.out}.ids\n" + \
		f"- {args.out}.win\n")

	 # Save haplotype cluster medians to binary file format
	if args.medians:
		print("Saved haplotype cluster medians in binary format:\n" + \
			f"- {args.out}.bcm\n" + \
			f"- {args.out}.blk\n" + \
			f"- {args.out}.wix\n")
	del e_vec, w_mat

	# Save haplotype cluster assignments in binary PLINK format
	if args.plink:
		print("\rGenerating binary PLINK output.", end="")
		K_tot = np.sum(k_vec, dtype=np.uint32)
		P_mat = np.zeros((K_tot, 3), dtype=np.uint32)
		Z_bin = np.zeros((K_tot, B), dtype=np.uint8)
		c_vec = np.insert(np.cumsum(k_vec[:-1], dtype=np.uint32), 0, 0)
		reader_cy.convertPlink(Z, Z_bin, P_mat, k_vec, c_vec, b_vec)
		
		# Save .bed file including magic numbers
		with open(f"{args.out}.bed", "w") as bfile:
			np.array([108, 27, 1], dtype=np.uint8).tofile(bfile)
			Z_bin.tofile(bfile)
		del b_vec, c_vec, Z_bin, Z

		# Save .bim file
		tmp = np.array([f"{chrom}_W{w}_K{k}_B{l}" for w,k,l in P_mat])
		bim = np.hstack((
			np.array([chrom]).repeat(K_tot).reshape(-1, 1),
			tmp.reshape(-1, 1),
			np.zeros((K_tot, 1), dtype=np.uint32),
			s_vec.repeat(k_vec).reshape(-1, 1),
			np.array(["K"]).repeat(K_tot).reshape(-1, 1),
			np.zeros((K_tot, 1), dtype=np.uint32)
		))
		np.savetxt(f"{args.out}.bim", bim, fmt="%s", delimiter="\t")
		del k_vec, s_vec, bim, tmp, P_mat

		# Save .fam file
		if args.duplicate_fid:
			s_list = s_list.repeat(2, axis=1)
		else:
			s_list = np.hstack((np.zeros((N//2, 1), dtype=np.uint8), s_list))
		fam = np.hstack((
			s_list,
			np.zeros((N//2, 3), dtype=np.uint8),
			np.full((N//2, 1), -9, dtype=np.int8)
		))
		np.savetxt(f"{args.out}.fam", fam, fmt="%s", delimiter="\t")

		# Print info
		print("\rSaved haplotype clusters in binary PLINK format:\n" + \
			f"- {args.out}.bed\n" + \
			f"- {args.out}.bim\n" + \
			f"- {args.out}.fam\n")
		del fam, s_list

	# Print elapsed time for parsing and total computation
	t_min = int(t_par//60)
	t_sec = int(t_par - t_min*60)
	print(f"Total parsing time: {t_min}m{t_sec}s")
	t_tot = time() - start
	t_min = int(t_tot//60)
	t_sec = int(t_tot - t_min*60)
	print(f"Total elapsed time: {t_min}m{t_sec}s")

	# Write to log-file
	with open(f"{args.out}.log", "a") as log:
		log.write("\nSaved haplotype clusters in binary format:\n"
			f"- {args.out}.bca\n" + \
			f"- {args.out}.ids\n" + \
			f"- {args.out}.win\n")
		if args.medians:
			log.write("\nSaved haplotype cluster medians in binary format:\n" + \
				f"- {args.out}.bcm\n" + \
				f"- {args.out}.blk\n" + \
				f"- {args.out}.wix\n")
		if args.plink:
			log.write("\nSaved haplotype clusters in binary PLINK format:\n" + \
				f"- {args.out}.bed\n" + \
				f"- {args.out}.bim\n" + \
				f"- {args.out}.fam\n")
		log.write(f"\nTotal elapsed time: {t_min}m{t_sec}s\n")



##### Main exception #####
assert __name__ != "__main__", "Please use the 'hapla cluster' command!"
