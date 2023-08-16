"""
hapla.
Haplotype clustering using K-Medians with delayed cluster creation.
"""

__author__ = "Jonas Meisner"

# Libraries
import os

##### hapla cluster #####
def main(args):
	print("hapla by Jonas Meisner (v0.2)")
	print(f"hapla cluster using {args.threads} thread(s).")
	
	# Check input
	assert args.vcf is not None, \
		"Please provide phased genotype file (--bcf or --vcf)!"
	assert args.min_freq > 0.0, "Empty haplotype clusters not allowed!"

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
	G = reader_cy.readVCF(v_file, n//2, B)
	del v_file
	m = G.shape[0]
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
		reader_cy.filterSNPs(G, winList, mask) # Fix data and window arrays
		G = G[:m,:]
		print(f"Removed {np.sum(mask==0)} causal SNPs.")
		del mask

	### Containers
	c_vec = np.zeros(n, dtype=np.int32) # Cost vector
	K_vec = np.zeros(W, dtype=np.uint8) # Number of clusters in windows
	N_vec = np.zeros(args.max_clusters, dtype=np.int32) # Size vector
	Z_mat = np.zeros((W, n), dtype=np.uint8) # Haplotype cluster alleles
	if args.windows is None:
		H = np.zeros((args.fixed, n), dtype=np.uint8) # Haplotypes
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

		# Load haplotype window
		if w < (W-1):
			if args.windows is None: # Re-use containers
				M.fill(-9)
				C.fill(0.0)
				reader_cy.convertBit(G, H, C, winList[w], args.threads)
			else:
				H = np.zeros((winList[w+1]-winList[w], n), dtype=np.uint8)
				M = np.full((args.max_clusters, H.shape[0]), -9, dtype=np.int8)
				C = np.zeros((args.max_clusters, H.shape[0]), dtype=np.float32)
				reader_cy.convertBit(G, H, C, winList[w], args.threads)
		else: # Last window
			H = np.zeros((m-winList[w], n), dtype=np.uint8)
			M = np.full((args.max_clusters, H.shape[0]), -9, dtype=np.int8)
			C = np.zeros((args.max_clusters, H.shape[0]), dtype=np.float32)
			reader_cy.convertBit(G, H, C, winList[w], args.threads)
		mX = H.shape[0]

		# Setup log-likelihood container
		if args.loglike:
			L_mat[w,:,:].fill(-16*mX) # Approximate -log(1e-7)*m

		# Compute mean and initialize first median
		K = 1
		N_vec[0] = n
		cluster_cy.marginalMedians(M, C, N_vec, K)
		X = np.ascontiguousarray(H.T) # Re-order NxB in contiguous memory
		if args.windows is not None:
			del H

		# Perform DC-DP-Medians
		for it in np.arange(args.max_iterations):
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
			if it > 0:
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
				print(f"Epoch {it}: Cost {cost}")
			# Count sizes and construct marginal medians
			cluster_cy.countN(Z_mat, N_vec, K, w)
			cluster_cy.marginalMedians(M, C, N_vec, K)

		# Remove small haplotype clusters and rescue as many as possible
		if K > 2:
			N_thr = int(args.min_freq*n)
			K_tmp = K
			if args.verbose:
				N_sur = np.sum(N_vec > N_thr)
				print(f"{N_sur}/{K_tmp} clusters reaching threshold.")
			while True:
				N_min = cluster_cy.findZero(N_vec, n, N_thr, K)
				if N_min > N_thr:
					break
				cluster_cy.clusterAssignment(X, M, C, Z_mat, c_vec, N_vec, K, w, \
					args.threads)
				cluster_cy.countN(Z_mat, N_vec, K, w)
				cluster_cy.marginalMedians(M, C, N_vec, K)
				K_tmp -= 1
				if K_tmp == 2: # Safety break
					break
				if args.verbose:
					N_sur = np.sum(N_vec > N_thr)
					print(f"{N_sur}/{K_tmp} clusters reaching threshold. {N_min}/{N_thr}.")
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
			else: # Last window
				M_mat[winList[w]:m] = np.ascontiguousarray(M.T)
		if args.loglike:
			cluster_cy.loglikeHaplo(L_mat, X, C, Z_mat, N_vec, K, w, args.threads)
			
		# Clean up
		if args.windows is not None:
			del M, C
		del X
		N_vec.fill(0)
	if not args.verbose:
		print("")

	##### Save output #####
	np.save(f"{args.out}.z", Z_mat)
	print(f"Saved haplotype cluster alleles as {args.out}.z.npy")
	np.savetxt(f"{args.out}.num_clusters", K_vec, fmt="%i")
	print(f"Saved the number of clusters per window as {args.out}.num_clusters")
	if args.medians:
		np.save(f"{args.out}.medians", M_mat)
		print(f"Saved haplotype cluster medians as {args.out}.medians.npy")
		del M_mat
	if args.loglike:
		np.save(f"{args.out}.loglike", L_mat)
		print(f"Saved haplotype cluster log-likelihoods as {args.out}.loglike.npy")
		del L_mat
	if args.plink:
		print("\rGenerating binary PLINK output.", end="")
		v_file = VCF(args.vcf, threads=args.threads)
		s_list = v_file.samples
		for variant in v_file: # Extract chromosome name from first entry
			chrom = np.array([variant.CHROM])
			break
		del v_file
		K_vec -= 1 # Dummy encoding
		K_tot = np.sum(K_vec, dtype=int)
		Z_vec = np.zeros(n//2, dtype=np.uint8)
		Z_bin = np.zeros((np.sum(K_vec, dtype=int), B), dtype=np.uint8)
		reader_cy.convertPlink(Z_mat, Z_bin, Z_vec, K_vec)

		# Save .bim file
		pos = np.zeros((K_tot, 2), dtype=int)
		reader_cy.clusterID(Z_mat, pos, K_vec)
		tmp = [f"{w}_{k}" for w,k in pos]
		bim = np.hstack((chrom.repeat(K_tot), tmp, np.zeros(K_tot, dtype=int), \
			np.arange(1, K_tot+1), np.array(["A"]).repeat(K_tot), \
			np.array(["T"]).repeat(K_tot)))
		np.savetxt(f"{args.out}.bim", bim, delimiter="\t", fmt="%s")
		del bim, pos, tmp

		# Save .bed file
		with open(f"{args.out}.bed", "w") as bfile:
			np.array([108,27,1], dtype=np.uint8).tofile(bfile) # Magic numbers
			Z_bin.tofile(bfile)
		del K_vec, Z_vec, Z_mat, Z_bin
		
		# Save .fam file
		tmp = np.zeros((n//2, 4), dtype=int)
		tmp[:,3] = -9
		fam = np.hstack((np.zeros(n//2, dtype=int), np.array(s_list), tmp))
		np.savetxt(f"{args.out}.fam", fam, delimiter="\t", fmt="%s")
		print(f"\rSaved haplotype cluster alleles in binary PLINK format as " + \
			f"{args.out}.(bed,bim,fam)")



##### Main exception #####
assert __name__ != "__main__", "Please use the 'hapla cluster' command!"
