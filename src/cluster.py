"""
hapla.
Haplotype clustering using K-Medians with delayed cluster creation.
"""

__author__ = "Jonas Meisner"

# Libraries
import os

##### hapla cluster #####
def main(args):
	print("hapla cluster by Jonas Meisner (v0.1)")
	print(f"Using {args.threads} thread(s).")
	
	# Check input
	assert args.vcf is not None, \
		"Please provide phased genotype file (--bcf or --vcf)!"
	assert args.min_count > 0, "Empty haplotype clusters not allowed!"

	# Control threads of external numerical libraries
	os.environ["MKL_NUM_THREADS"] = str(args.threads)
	os.environ["OMP_NUM_THREADS"] = str(args.threads)
	os.environ["NUMEXPR_NUM_THREADS"] = str(args.threads)
	os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)

	# Import numerical libraries and cython functions
	import numpy as np
	from cyvcf2 import VCF
	from math import ceil
	from src import reader_cy
	from src import cluster_cy

	### Load data into 1-bit matrix
	print("\rLoading VCF/BCF file...", end="")
	v_file = VCF(args.vcf, threads=args.threads)
	n = 2*len(v_file.samples)
	B = ceil(n/8)
	Gt = reader_cy.readVCF(v_file, n//2, B)
	del v_file
	m = Gt.shape[0]
	print(f"\rLoaded phased genotype data: {n} haplotypes and {m} SNPs.")

	### Setup windows
	if args.windows is None: # Fixed window length
		W = m//args.fixed
		winList = [w*args.fixed for w in range(W)]
		winList = np.array(winList, dtype=int)
		print(f"Clustering in {W} windows of fixed size ({args.fixed} SNPs).")
	else: # Use provided window lengths
		winList = np.genfromtxt(args.windows, dtype=int)
		W = winList.shape[0] - 1
		assert winList[-1] == m, "Window splits doesn't match genotype file!"
		print(f"Clustering in {W} windows of provided lengths.")
	
		# Filter out causal SNPs --- DEBUG FOR SIMULATION STUDIES ONLY!
		if args.filter is not None:
			mask = np.loadtxt(args.filter, dtype=np.uint8)
			m = np.sum(mask) # New number of variants
			reader_cy.filterSNPs(Gt, winList, mask) # Fix data and window arrays
			Gt = Gt[:m,:]
			print(f"Removed {np.sum(mask==0)} causal SNPs.")
			del mask

	### Containers
	c_vec = np.zeros(n, dtype=np.int32) # Cost vector
	K_vec = np.zeros(W, dtype=np.uint8) # Number of clusters in windows
	N_vec = np.zeros(args.max_clusters, dtype=np.int32) # Size vector
	Z_mat = np.zeros((W, n), dtype=np.uint8) # Haplotype cluster assignments
	if args.windows is None:
		M = np.zeros((args.max_clusters, args.fixed), dtype=np.int8) # Medians
		C = np.zeros((args.max_clusters, args.fixed), dtype=np.float32) # Means
	if args.medians:
		M_mat = np.zeros((m, args.max_clusters), dtype=np.int8)
	if args.loglike:
		L_mat = np.zeros((W, n, args.max_clusters), dtype=np.float32)

	##### Clustering using DC-DP-Medians #####
	for w in np.arange(W):
		if args.verbose:
			print(f"Window {w+1}/{W}")
		else:
			print(f"\rWindow {w+1}/{W}", end="")

		# Load haplotype segment
		if w < (W-1):
			if args.windows is None:
				Xt = np.zeros((args.fixed, n), dtype=np.uint8)
				M.fill(-9)
				C.fill(0.0)
				reader_cy.convertBit(Gt, Xt, C, winList[w], args.threads)
			else:
				Xt = np.zeros((winList[w+1]-winList[w], n), dtype=np.uint8)
				M = np.full((args.max_clusters, Xt.shape[0]), -9, dtype=np.int8)
				C = np.zeros((args.max_clusters, Xt.shape[0]), dtype=np.float32)
				reader_cy.convertBit(Gt, Xt, C, winList[w], args.threads)
		else:
			Xt = np.zeros((m-winList[w], n), dtype=np.uint8)
			M = np.full((args.max_clusters, Xt.shape[0]), -9, dtype=np.int8)
			C = np.zeros((args.max_clusters, Xt.shape[0]), dtype=np.float32)
			reader_cy.convertBit(Gt, Xt, C, winList[w], args.threads)
		mX = Xt.shape[0]

		# Setup log-likelihood container
		if args.loglike:
			L_mat[w,:,:].fill(-16*mX) # Approximate -log(1e-7)*m

		# Compute mean and initialize first median
		K = 1
		N_vec[0] = n
		cluster_cy.marginalMedians(M, C, N_vec, K)
		X = np.ascontiguousarray(Xt.T)
		del Xt

		# Perform DC-DP-Medians
		for iter in np.arange(args.max_iterations):
			# Cluster assignment
			cluster_cy.clusterAssignment(X, M, C, Z_mat, c_vec, N_vec, K, w, \
				args.threads)

			# Check for extra cluster
			c_max = np.max(c_vec)
			c_arg = np.argmax(c_vec)
			if (c_max > args.lmbda*mX) & (K < args.max_clusters):
				M[K,:] = X[c_arg,:]
				C[K,:] = X[c_arg,:]
				C[Z_mat[w,c_arg],:] -= X[c_arg,:]
				Z_mat[w,c_arg] = K
				K += 1

			# Check for convergence
			if iter > 0:
				if np.allclose(Z_mat[w], z_prev):
					if K > 1:
						# Count sizes and construct marginal medians
						cluster_cy.countN(Z_mat, N_vec, K, w)
						cluster_cy.marginalMedians(M, C, N_vec, K)
						if args.verbose:
							print("Converged! No label switching.")
						break
					else: # Make sure two haplotype clusters are generated
						M[K,:] = X[c_arg,:]
						C[K,:] = X[c_arg,:]
						C[Z_mat[w,c_arg],:] -= X[c_arg,:]
						Z_mat[w,c_arg] = K
						K += 1
			z_prev = np.copy(Z_mat[w])
			if args.verbose:
				cost = np.sum(c_vec) + args.lmbda*mX*K
				print(f"Epoch {iter}: Cost {cost}")
			# Count sizes and construct marginal medians
			cluster_cy.countN(Z_mat, N_vec, K, w)
			cluster_cy.marginalMedians(M, C, N_vec, K)

		# Remove small haplotype clusters and try to rescue as many as possible
		if K > 2:
			N_sur = max(2, np.sum(N_vec >= args.min_count)) # Surviving clusters
			N_tmp = N_sur
			K_rem = K - N_sur
			for k in np.arange(K_rem):
				cluster_cy.findZero(N_vec, n, args.min_count, K) # Smallest cluster
				cluster_cy.clusterAssignment(X, M, C, Z_mat, c_vec, N_vec, K, w, \
					args.threads)
				cluster_cy.countN(Z_mat, N_vec, K, w)
				N_sur = np.sum(N_vec > 0)
				if N_sur == N_tmp:
					break
				if k < (K_rem - 1):
					cluster_cy.marginalMedians(M, C, N_vec, K)
				N_tmp = N_sur
		else:
			cluster_cy.clusterAssignment(X, M, C, Z_mat, c_vec, N_vec, K, w, \
				args.threads)
			cluster_cy.countN(Z_mat, N_vec, K, w)

		# Fix cluster median and cluster assignment order
		cluster_cy.medianFix(M, Z_mat, N_vec, K, w)
		K = np.sum(N_vec > 0)
		K_vec[w] = K

		# Generate optional saves (medians and log-likehoods)
		if args.medians:
			if w < (W - 1):
				if args.windows is None:
					M_mat[winList[w]:(winList[w]+args.fixed)] = \
						np.ascontiguousarray(M.T)
				else:
					M_mat[winList[w]:winList[w+1]] = \
						np.ascontiguousarray(M.T)
			else:
				M_mat[winList[w]:m] = np.ascontiguousarray(M.T)
		if args.loglike:
			cluster_cy.loglikeHaplo(L_mat, X, C, Z_mat, N_vec, K, w, args.threads)
			
		# Clean up
		del X
		N_vec.fill(0)
	if not args.verbose:
		print("")

	##### Save output #####
	np.save(f"{args.out}.z", Z_mat)
	print(f"Saved haplotype cluster assignments as {args.out}.z.npy")
	np.savetxt(f"{args.out}.num_clusters", K_vec, fmt="%i")
	print(f"Saved the number of clusters per window as {args.out}.num_clusters")
	if args.medians:
		np.save(f"{args.out}.medians", M_mat)
		print(f"Saved haplotype cluster medians as {args.out}.medians.npy")
	if args.loglike:
		np.save(f"{args.out}.loglike", L_mat)
		print(f"Saved haplotype cluster log-likelihoods as {args.out}.loglike.npy")



##### Main exception #####
assert __name__ != "__main__", "Please use the 'hapla cluster' command!"
