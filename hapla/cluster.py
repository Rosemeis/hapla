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
	print("--------------------------------")
	print("hapla by Jonas Meisner (v0.3)")
	print(f"hapla cluster using {args.threads} thread(s)")
	print("--------------------------------\n")
	
	# Check input
	assert args.vcf is not None, \
		"Please provide phased genotype file (--bcf or --vcf)!"
	assert args.min_freq > 0.0, "Empty haplotype clusters not allowed!"
	assert args.max_clusters <= 256, "Max clusters allowed exceeded!"
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

	### Load data into 1-bit matrix
	print("\rLoading VCF/BCF file...", end="")
	v_file = VCF(args.vcf, threads=min(args.threads, 4))
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
	z_pre = np.zeros(n, dtype=np.uint8) # Help vector
	K_vec = np.zeros(W, dtype=np.uint8) # Number of clusters in windows
	N_vec = np.zeros(args.max_clusters, dtype=np.int32) # Size vector
	Z_mat = np.zeros((W, n), dtype=np.uint8) # Haplotype cluster alleles
	if args.windows is None:
		H = np.zeros((args.fixed, n), dtype=np.uint8) # Haplotypes
		Ht = np.zeros((n, args.fixed), dtype=np.uint8) # Haplotypes transposed
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
				reader_cy.convertBit(G, H, C, winList[w])
			else:
				H = np.zeros((winList[w+1]-winList[w], n), dtype=np.uint8)
				Ht = np.zeros((n, winList[w+1]-winList[w]), dtype=np.uint8)
				M = np.full((args.max_clusters, H.shape[0]), -9, dtype=np.int8)
				C = np.zeros((args.max_clusters, H.shape[0]), dtype=np.float32)
				reader_cy.convertBit(G, H, C, winList[w])
		else: # Last window
			H = np.zeros((m-winList[w], n), dtype=np.uint8)
			Ht = np.zeros((n, m-winList[w]), dtype=np.uint8)
			M = np.full((args.max_clusters, H.shape[0]), -9, dtype=np.int8)
			C = np.zeros((args.max_clusters, H.shape[0]), dtype=np.float32)
			reader_cy.convertBit(G, H, C, winList[w])
		mH = H.shape[0]

		# Setup log-likelihood container
		if args.loglike:
			L_mat[w,:,:].fill(-16*mH) # Approximate -log(1e-7)*m

		# Compute mean and initialize first median
		K = 1
		N_vec[0] = n
		cluster_cy.marginalMedians(M, C, N_vec, K)
		np.copyto(Ht, H.T, casting="no") # Transposed in contiguous memory
		if args.windows is not None:
			del H

		# Perform DC-DP-Medians
		for it in np.arange(args.max_iterations):
			np.copyto(z_pre, Z_mat[w], casting="no")

			# Cluster assignment
			cluster_cy.clusterAssignment(Ht, M, C, Z_mat, c_vec, N_vec, K, w, \
				args.threads)

			# Check for extra cluster
			c_max = np.max(c_vec)
			c_arg = np.argmax(c_vec)
			if (c_max > args.lmbda*mH) & (K < args.max_clusters):
				M[K,:] = Ht[c_arg,:]
				C[K,:] = Ht[c_arg,:]
				C[Z_mat[w,c_arg],:] -= Ht[c_arg,:]
				Z_mat[w,c_arg] = K
				K += 1

			# Check for convergence
			if it > 0:
				if np.allclose(Z_mat[w], z_pre):
					if K > 1:
						# Count sizes and construct marginal medians
						cluster_cy.countN(Z_mat, N_vec, K, w)
						cluster_cy.marginalMedians(M, C, N_vec, K)
						if args.verbose:
							print("Converged! No label switching.")
						break
					else: # Make sure two haplotype clusters are generated
						M[K,:] = Ht[c_arg,:]
						C[K,:] = Ht[c_arg,:]
						C[Z_mat[w,c_arg],:] -= Ht[c_arg,:]
						Z_mat[w,c_arg] = K
						K += 1
			if args.verbose:
				cost = np.sum(c_vec) + args.lmbda*mH*K
				print(f"Epoch {it}: Cost {cost}")
			# Count sizes and construct marginal medians
			cluster_cy.countN(Z_mat, N_vec, K, w)
			cluster_cy.marginalMedians(M, C, N_vec, K)

		# Remove small haplotype clusters and rescue as many as possible
		if K > 2:
			# Remove up and including to doubletons
			N_thr = max(2, int(args.min_freq*n))
			N_vec[N_vec <= min(2, N_thr)] = 0
			cluster_cy.clusterAssignment(Ht, M, C, Z_mat, c_vec, N_vec, K, w, \
				args.threads)
			cluster_cy.countN(Z_mat, N_vec, K, w)
			cluster_cy.marginalMedians(M, C, N_vec, K)
			K_tmp = np.sum(N_vec > 0)

			# Remove small clusters iterativly
			if args.verbose:
				N_sur = np.sum(N_vec > N_thr)
				print(f"{N_sur}/{K_tmp} clusters reaching threshold.")
			while True:
				N_min = cluster_cy.findZero(N_vec, n, N_thr, K)
				if N_min > N_thr:
					break
				cluster_cy.clusterAssignment(Ht, M, C, Z_mat, c_vec, N_vec, K, w, \
					args.threads)
				cluster_cy.countN(Z_mat, N_vec, K, w)
				cluster_cy.marginalMedians(M, C, N_vec, K)
				K_tmp -= 1
				if K_tmp == 2: # Safety break
					break
				if args.verbose:
					N_sur = np.sum(N_vec > N_thr)
					print(f"{N_sur}/{K_tmp} clusters reaching threshold. " + \
						f"{N_min}/{N_thr}.")
		else:
			cluster_cy.clusterAssignment(Ht, M, C, Z_mat, c_vec, N_vec, K, w, \
				args.threads)
			cluster_cy.countN(Z_mat, N_vec, K, w)

		# Fix cluster median and cluster assignment order
		cluster_cy.medianFix(M, Z_mat, N_vec, K, w)
		K = np.sum(N_vec > 0)
		K_vec[w] = K

		# Generate optional saves (medians and log-likehoods)
		if args.medians:
			if w < (W-1):
				if args.windows is None:
					M_mat[winList[w]:(winList[w]+args.fixed)] = \
						np.ascontiguousarray(M.T)
				else:
					M_mat[winList[w]:winList[w+1]] = \
						np.ascontiguousarray(M.T)
			else: # Last window
				M_mat[winList[w]:m] = np.ascontiguousarray(M.T)
		if args.loglike:
			cluster_cy.loglikeHaplo(L_mat, Ht, C, Z_mat, N_vec, K, w, args.threads)
			
		# Clean up
		if args.windows is not None:
			del M, C, Ht
		N_vec.fill(0)
	del G
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
		del M_mat
	if args.loglike:
		np.save(f"{args.out}.loglike", L_mat)
		print(f"Saved haplotype cluster log-likelihoods as {args.out}.loglike.npy")
		del L_mat
	if args.plink:
		print("\rGenerating binary PLINK output.", end="")
		import re
		v_file = VCF(args.vcf, threads=min(args.threads, 4))
		s_list = np.array(v_file.samples).reshape(-1,1)
		for variant in v_file: # Extract chromosome name from first entry
			chrom = re.findall(r'\d+', variant.CHROM)[-1]
			break
		del v_file
		K_tot = np.sum(K_vec, dtype=int)
		P_mat = np.zeros((K_tot, 2), dtype=np.int32)
		Z_vec = np.zeros(n//2, dtype=np.uint8)
		Z_bin = np.zeros((K_tot, B), dtype=np.uint8)
		reader_cy.convertPlink(Z_mat, Z_bin, P_mat, Z_vec, K_vec)
		
		# Save .bed file including magic numbers
		with open(f"{args.out}.bed", "w") as bfile:
			np.array([108,27,1], dtype=np.uint8).tofile(bfile)
			Z_bin.tofile(bfile)
		del K_vec, Z_bin, Z_mat, Z_vec

		# Save .bim file
		tmp = np.array([f"{chrom}_{w}_{k}" for w,k in P_mat]).reshape(-1,1)
		bim = np.hstack((np.array([chrom]).repeat(K_tot).reshape(-1,1), \
			tmp, np.zeros((K_tot, 1), dtype=np.uint8), \
			np.arange(1, K_tot+1).reshape(-1,1), \
			np.array(["A"]).repeat(K_tot).reshape(-1,1), \
			np.array(["T"]).repeat(K_tot).reshape(-1,1)))
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
assert __name__ != "__main__", "Please use the 'hapla cluster' command!"
