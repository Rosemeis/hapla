"""
hapla.
Population structure inference using haplotype cluster alleles.
"""

__author__ = "Jonas Meisner"

# Libraries
import os
from time import time

##### hapla struct #####
def main(args):
	print("-----------------------------------")
	print("hapla by Jonas Meisner (v0.8)")
	print(f"hapla struct using {args.threads} thread(s)")
	print("-----------------------------------\n")

	# Check input
	assert (args.filelist is not None) or (args.clusters is not None), \
		"No input data (--filelist or --clusters)!"
	assert (args.grm or (args.pca is not None)), \
		"No analysis selected (--grm, --pca)!"
	if args.grm:
		assert args.iid is not None, "Provide sample list for GCTA format (--iid)!"
	if args.pca is not None:
		assert args.pca > 0, "Please select a valid number of eigenvectors!"
	start = time()

	# Control threads of external numerical libraries
	os.environ["MKL_NUM_THREADS"] = str(args.threads)
	os.environ["OMP_NUM_THREADS"] = str(args.threads)
	os.environ["NUMEXPR_NUM_THREADS"] = str(args.threads)
	os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)

	# Import numerical libraries and cython functions
	import numpy as np
	from math import ceil
	from scipy.sparse.linalg import svds
	from hapla import functions
	from hapla import shared_cy

	# Load data (and concatentate across windows)
	if args.filelist is not None:
		Z_list = []
		with open(args.filelist) as f:
			for z_file in f:
				Z_list.append(z_file.strip("\n"))
	else:
		Z_list = [args.clusters]
	print(f"Parsing {len(Z_list)} file(s).")
	
	# Estimate genome-wide relationship matrix
	if args.grm:
		print("Estimating genome-wide relationship matrix (GRM).")
		M = 0
		s_pre = ""
		for z in np.arange(len(Z_list)): # Loop through files
			# Print information
			s_beg = f"Processing file {z+1}/{len(Z_list)}"
			print(f"\r{s_beg: <{len(s_pre)}}", end="")

			# Load file
			Z_mat = np.load(Z_list[z])
			if z == 0:
				n = Z_mat.shape[1]//2
				G = np.zeros((n, n), dtype=np.float32)
			else:
				assert n == Z_mat.shape[1]//2, \
					"Number of samples differ between cluster files!"
			W = Z_mat.shape[0]

			# Count haplotype cluster alleles and find rarest clusters
			k_vec = np.max(Z_mat, axis=1) + 1
			m = np.sum(k_vec, dtype=int)

			# Populate full matrix and estimate cluster frequencies
			Z = np.zeros((m, n), dtype=np.uint8)
			p = np.zeros(m, dtype=np.float32)
			shared_cy.haplotypeAggregate(Z_mat, Z, p, k_vec)
			del Z_mat, k_vec

			# Setup GRM part settings
			B = ceil(m/args.batch) # Number of batches
			a = np.ones(m, dtype=np.float32)
			Z_b = np.zeros((args.batch, n), dtype=np.float32)

			# Estimate GRM part in batches
			for b in np.arange(B):
				s_bat = f"{s_beg}. Batch {b+1}/{B}"
				print(f"\r{s_bat}", end="") # Print information
				m_b = b*args.batch
				if b == (B-1): # Last batch
					Z_b = np.zeros((m - m_b, n), dtype=np.float32)
				shared_cy.batchZ(Z, Z_b, p, a, m_b, args.threads)

				# Aggregate across batches
				G += np.dot(Z_b.T, Z_b)
			M += m
			s_pre = s_bat
			del a, p, Z, Z_b
		G *= (1.0/M)
		print(".\n")
		
		# Centering
		if not args.no_centering:
			print("Centering GRM.")
			u = np.mean(G, axis=1)
			G -= u.reshape(1, n)
			u = np.mean(G, axis=1)
			G -= u.reshape(n, 1)

			# Gower centering
			G *= float(n-1)/np.trace(G)
			del u

		# Save matrix
		G = G[np.tril_indices(n)]
		G.tofile(f"{args.out}.grm.bin")
		np.full(n*(n+1)//2, M, dtype=np.float32).tofile(f"{args.out}.grm.N.bin")
		iid = np.loadtxt(f"{args.iid}", dtype=np.str_).reshape(-1,1)
		if args.fid is not None:
			fid = np.loadtxt(f"{args.fid}", dtype=np.str_).reshape(-1,1)
			fam = np.hstack((fid, iid))
			del fid
		else:
			fam = np.hstack((np.zeros((n, 1), dtype=np.uint8), iid))
		np.savetxt(f"{args.out}.grm.id", fam, delimiter="\t", fmt="%s")
		print("Saved genome-wide relationship matrix in GCTA format:\n" + \
			f"- {args.out}.grm.bin\n" + \
			f"- {args.out}.grm.N.bin\n" + \
			f"- {args.out}.grm.id\n")
		del G, iid, fam
	
	# Infer population structure using PCA
	if args.pca is not None:
		# Load and merge haplotype cluster alleles
		Z_tmp = []
		for z in np.arange(len(Z_list)):
			Z_tmp.append(np.load(Z_list[z]))
			print(f"\rParsed file {z+1}/{len(Z_list)}", end="")
		Z_mat = np.concatenate(Z_tmp, axis=0)
		del Z_tmp
		W = Z_mat.shape[0]
		n = Z_mat.shape[1]//2

		# Count haplotype cluster alleles
		k_vec = np.max(Z_mat, axis=1) + 1
		m = np.sum(k_vec, dtype=int)

		# Print information
		print(f"\rLoaded haplotype cluster assignments:\n" + \
			f"- {n} samples\n" + \
			f"- {W} windows\n" + \
			f"- {m} clusters\n")
		
		# Populate full matrix and estimate cluster frequencies
		Z = np.zeros((m, n), dtype=np.uint8)
		p = np.zeros(m, dtype=np.float32)
		shared_cy.haplotypeAggregate(Z_mat, Z, p, k_vec)
		del Z_mat, k_vec
		a = np.power(2.0*p*(1-p), args.scaling)

		# Randomized SVD
		if args.randomized:
			print(f"Computing randomized SVD, extracting {args.pca} eigenvectors.")

			# Randomized SVD in batches
			U, S, V = functions.randomizedSVD(Z, p, a, args.pca, args.batch, \
				args.threads)

			# Save matrices
			if args.iid is not None:
				iid = np.loadtxt(f"{args.iid}", dtype=np.str_).reshape(-1,1)
				if args.fid is not None:
					fid = np.loadtxt(f"{args.fid}", dtype=np.str_).reshape(-1,1)
					fam = np.hstack((fid, iid))
				else:
					fam = np.hstack((np.zeros((n, 1), dtype=np.uint8), iid))
				V = np.hstack((fam, np.round(V, 7)))
			np.savetxt(f"{args.out}.eigenvec", V, fmt="%.7f")
			np.savetxt(f"{args.out}.eigenval", (S*S)/float(m), fmt="%.7f")
			if args.loadings:
				np.savetxt(f"{args.out}.loadings", U, fmt="%.7f")
				print(f"Saved loadings as {args.out}.loadings")
		else: # Truncated SVD
			print(f"Computing truncated SVD, extracting {args.pca} eigenvectors.")

			# Standardize
			Z_s = np.zeros((m, n), dtype=np.float32)
			shared_cy.standardizeZ(Z, Z_s, p, a, args.threads)

			# Truncated SVD (Arnoldi)
			U, S, Vt = svds(Z_s, k=args.pca)
			del Z_s

			# Save matrices
			if args.iid is not None:
				iid = np.loadtxt(f"{args.iid}", dtype=np.str_).reshape(-1,1)
				if args.fid is not None:
					fid = np.loadtxt(f"{args.fid}", dtype=np.str_).reshape(-1,1)
					fam = np.hstack((fid, iid))
				else:
					fam = np.hstack((np.zeros((n, 1), dtype=np.uint8), iid))
				out = np.hstack((fam, np.round(Vt[::-1,:].T, 7)))
				np.savetxt(f"{args.out}.eigenvec", out, fmt="%s")
			else:
				np.savetxt(f"{args.out}.eigenvec", Vt[::-1,:].T, fmt="%.7f")
			np.savetxt(f"{args.out}.eigenval", (S[::-1]*S[::-1])/float(m), fmt="%.7f")
			if args.loadings:
				np.savetxt(f"{args.out}.loadings", U[:,::-1], fmt="%.7f")
				print(f"Saved loadings as {args.out}.loadings")
		print(f"Saved eigenvectors as {args.out}.eigenvec")
		print(f"Saved eigenvalues as {args.out}.eigenval\n")

	# Print elapsed time for computation
	t_tot = time()-start
	t_min = int(t_tot//60)
	t_sec = int(t_tot - t_min*60)
	print(f"Total elapsed time: {t_min}m{t_sec}s")



##### Main exception #####
assert __name__ != "__main__", "Please use the 'hapla struct' command!"
