"""
hapla.
Haplotype clustering using K-Medians with delayed cluster creation.
"""

__author__ = "Jonas Meisner"

# Libraries
import os
from time import time

##### hapla cluster #####
def main(args):
	print("-----------------------------------")
	print("hapla by Jonas Meisner (v0.6)")
	print(f"hapla cluster using {args.threads} thread(s)")
	print("-----------------------------------\n")
	
	# Check input
	assert args.vcf is not None, \
		"No phased genotype file (--bcf or --vcf)!"
	assert args.win > 0, "Invalid window size!"
	assert args.min_freq > 0.0, "Invalid haplotype cluster frequency!"
	assert args.max_clusters <= 256, "Max allowed clusters exceeded!"
	if args.overlap is not None:
		if args.win == 1:
			args.overlap = 0 
		assert (args.win % (args.overlap + 1) == 0), \
			"Invalid number of overlapping windows chosen!"
	else:
		args.overlap = 0
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
	from hapla import cluster_cy

	# Load data into 1-bit matrix
	print("\rLoading VCF/BCF file...", end="")
	v_file = VCF(args.vcf, threads=args.threads)
	m = 0
	n = 2*len(v_file.samples)
	B = ceil(n/8)

	# Check number of sites and allocate memory
	for variant in v_file:
		m += 1
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

	# Check haplotype cluster frequency
	N_mac = ceil(n*args.min_freq)
	assert N_mac > 2, "Frequency threshold too low for sample size (--min-freq)!"

	# Setup windows
	W = m//args.win
	if args.overlap > 0:
		W += (W - 1)*args.overlap
		w_vec = [w*(args.win//(args.overlap + 1)) for w in range(W)]
		print(f"Clustering {W} overlapping windows of {args.win} SNPs.")
	else:
		w_vec = [w*args.win for w in range(W)]
		print(f"Clustering {W} non-overlapping windows of {args.win} SNPs.")
	w_vec = np.array(w_vec, dtype=int)

	# Containers
	c_vec = np.zeros(n, dtype=np.int32) # Cost vector
	z_pre = np.zeros(n, dtype=np.uint8) # Help vector
	K_vec = np.zeros(W, dtype=np.uint8) # Number of clusters in windows
	N_vec = np.zeros(args.max_clusters, dtype=np.int32) # Size vector
	H = np.zeros((args.win, n), dtype=np.uint8) # Haplotypes
	X = np.zeros((n, args.win), dtype=np.uint8) # Haplotypes transposed
	M = np.zeros((args.max_clusters, args.win), dtype=np.int8) # Medians
	C = np.zeros((args.max_clusters, args.win), dtype=np.float32) # Means
	Z = np.zeros((W, n), dtype=np.uint8) # Haplotype cluster assignments

	# Thread-local containers
	I_thr = np.zeros((args.threads, 2), dtype=np.int32)
	N_thr = np.zeros((args.threads, args.max_clusters), dtype=np.int32)
	C_thr = np.zeros((args.threads, args.max_clusters, args.win), dtype=np.float32)
	for t in range(args.threads-1):
		I_thr[t] = [t*(n//args.threads), (t+1)*(n//args.threads)]
	I_thr[args.threads-1] = [(args.threads-1)*(n//args.threads), n]

	# Optional containers
	if args.medians:
		M_dict = {"I":np.array([m, args.win, W, args.overlap], dtype=int)}
	if args.loglike:
		L_dict = {}
		L = np.zeros((n, args.max_clusters), dtype=np.float32) # Log-likelihoods

	# Clustering using DC-DP-Medians
	for w in np.arange(W):
		if args.verbose:
			print(f"Window {w+1}/{W}")
		else:
			print(f"\rWindow {w+1}/{W}", end="")

		# Load haplotype window
		if w < (W-1):
			M.fill(-9)
		else: # Last window
			H = np.zeros((m-w_vec[w], n), dtype=np.uint8)
			X = np.zeros((n, m-w_vec[w]), dtype=np.uint8)
			M = np.full((args.max_clusters, H.shape[0]), -9, dtype=np.int8)
			C = np.zeros((args.max_clusters, H.shape[0]), dtype=np.float32)
			C_thr = np.zeros((args.threads, args.max_clusters, H.shape[0]), \
				dtype=np.float32)
		reader_cy.convertBit(G, H, C, w_vec[w])

		# Transposed in contiguous memory
		np.copyto(X, H.T, casting="no")

		# Compute mean and initialize first median
		K = 1
		N_vec[0] = n
		cluster_cy.marginalMedians(M, C, N_vec, K)

		# Perform DC-DP-Medians
		for it in np.arange(args.max_iterations):
			cluster_cy.clusterAssignment(X, M, Z, c_vec, N_vec, I_thr, N_thr, C_thr, \
				K, w, args.threads)
			np.sum(C_thr, axis=0, out=C)
			np.sum(N_thr, axis=0, out=N_vec)

			# Check for extra cluster
			c_max = np.max(c_vec)
			c_arg = np.argmax(c_vec)
			if (c_max > args.lmbda*H.shape[0]) & (K < args.max_clusters):
				M[K,:] = X[c_arg,:]
				C[K,:] = X[c_arg,:]
				C[Z[w,c_arg],:] -= X[c_arg,:]
				N_vec[K] = 1
				N_vec[Z[w,c_arg]] -= 1
				Z[w,c_arg] = K
				K += 1

			# Check for convergence
			if it > 0:
				if np.array_equal(Z[w], z_pre):
					if K > 1:
						if args.verbose:
							print(f"Converged! K={K}.")
						break
					else: # Make sure two haplotype clusters are generated
						print(" No diversity (K=1)! Adding extra cluster.")
						M[K,:] = X[c_arg,:]
						C[K,:] = X[c_arg,:]
						C[Z[w,c_arg],:] -= X[c_arg,:]
						N_vec[K] = 1
						N_vec[Z[w,c_arg]] -= 1
						Z[w,c_arg] = K
						K += 1
			if args.verbose:
				cost = np.sum(c_vec) + args.lmbda*H.shape[0]*K
				print(f"Epoch {it}: Cost {cost:.1f}")
			
			# Count sizes and construct marginal medians
			cluster_cy.marginalMedians(M, C, N_vec, K)
			np.copyto(z_pre, Z[w], casting="no")

		# Ensure correct medians
		cluster_cy.marginalMedians(M, C, N_vec, K)

		# Remove small haplotype clusters and rescue as many as possible
		if K > 2:
			# Remove singletons in one go
			N_vec[N_vec == 1] = 0
			cluster_cy.clusterAssignment(X, M, Z, c_vec, N_vec, I_thr, N_thr, C_thr, \
				K, w, args.threads)
			np.sum(C_thr, axis=0, out=C)
			np.sum(N_thr, axis=0, out=N_vec)
			K_tmp = np.sum(N_vec > 0)

			# Remove small clusters iterativly
			if args.verbose:
				N_sur = np.sum(N_vec >= N_mac)
				print(f"{N_sur}/{K_tmp} clusters reaching threshold.")
			while K_tmp > 2:
				cluster_cy.marginalMedians(M, C, N_vec, K)

				# Find smallest cluster
				N_min = cluster_cy.findZero(N_vec, n, N_mac, K)
				if N_min >= N_mac:
					break
				K_tmp -= 1

				# Re-assign haplotypes
				cluster_cy.clusterAssignment(X, M, Z, c_vec, N_vec, I_thr, N_thr, \
					C_thr, K, w, args.threads)
				np.sum(C_thr, axis=0, out=C)
				np.sum(N_thr, axis=0, out=N_vec)
				if args.verbose:
					N_sur = np.sum(N_vec >= N_mac)
					print(f"{N_sur}/{K_tmp} clusters reaching threshold. " + \
						f"{N_min}/{N_mac}.")

		# Fix cluster median and cluster assignment order
		cluster_cy.medianFix(M, Z, N_vec, K, w, args.threads)
		K = np.sum(N_vec > 0, dtype=int)
		K_vec[w] = K
		if args.plink:
			R_vec[w] = np.argmin(N_vec[:K])

		# Generate optional saves (medians and log-likehoods)
		if args.medians:
			M_dict[f"W{w}"] = M[:K].copy()
		if args.loglike:
			cluster_cy.loglikeHaplo(L, X, C, Z, N_vec, K, w, args.threads)
			L_dict[f"W{w}"] = L[:,:K].copy()
			
		# Clean up
		C_thr.fill(0)
		N_thr.fill(0)
		N_vec.fill(0)		
	del G
	if not args.verbose:
		print(".\n")

	# Save output
	np.save(f"{args.out}.z", Z)
	np.savetxt(f"{args.out}.num_clusters", K_vec, fmt="%i")
	print(f"Saved haplotype cluster assignments as {args.out}.z.npy")
	print(f"Saved the number of clusters per window as {args.out}.num_clusters")
	if args.medians:
		np.savez(f"{args.out}.medians", **M_dict)
		print(f"Saved haplotype cluster medians as {args.out}.medians.npz")
		del M_dict
	if args.loglike:
		np.savez(f"{args.out}.loglike", **L_dict)
		print(f"Saved haplotype cluster log-likelihoods as {args.out}.loglike.npz")
		del L_dict
	if args.plink:
		print("\rGenerating binary PLINK output.", end="")
		import re
		v_file = VCF(args.vcf, threads=args.threads)
		s_list = np.array(v_file.samples).reshape(-1,1)
		for variant in v_file: # Extract chromosome name from first entry
			chrom = re.findall(r'\d+', variant.CHROM)[-1]
			break
		K_tot = np.sum(K_vec, dtype=int)
		P_mat = np.zeros((K_tot, 2), dtype=np.int32)
		Z_vec = np.zeros(n//2, dtype=np.uint8)
		Z_bin = np.zeros((K_tot, B), dtype=np.uint8)
		reader_cy.convertPlink(Z, Z_bin, P_mat, Z_vec, K_vec)
		
		# Save .bed file including magic numbers
		with open(f"{args.out}.bed", "w") as bfile:
			np.array([108, 27, 1], dtype=np.uint8).tofile(bfile)
			Z_bin.tofile(bfile)
		del K_vec, Z_bin, Z, Z_vec

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
