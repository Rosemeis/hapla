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
	print("hapla by Jonas Meisner (v0.12)")
	print(f"hapla struct using {args.threads} thread(s)")
	print("-----------------------------------\n")

	# Check input
	assert (args.filelist is not None) or (args.clusters is not None), \
		"No input data (--filelist or --clusters)!"
	assert (args.grm or (args.pca is not None)), \
		"No analysis selected (--grm, --pca)!"
	if args.pca is not None:
		assert args.pca > 0, "Please select a valid number of eigenvectors!"
	assert args.threads > 0, "Please select a valid number of threads!"
	assert args.batch > 0, "Please select a valid batch size!"
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
	from math import ceil
	from scipy.sparse.linalg import svds
	from hapla import functions
	from hapla import shared_cy

	# Prepare list of data files
	if args.filelist is not None:
		W = 0 # Counter for windows
		Z_list = [] # List of filenames
		with open(args.filelist) as f:
			for z_file in f:
				# Check input across files and count windows
				z = z_file.strip("\n")
				Z_list.append(z)
				assert os.path.isfile(f"{z}.bca"), "bca file doesn't exist!"
				assert os.path.isfile(f"{z}.ids"), "ids file doesn't exist!"
				assert os.path.isfile(f"{z}.win"), "win file doesn't exist!"
				if W == 0: # First file
					z_ids = np.loadtxt(f"{z}.ids", dtype=np.str_)
					k_vec = np.loadtxt(f"{z}.win", dtype=np.uint8, usecols=[5])
					n = z_ids.shape[0]
					W = k_vec.shape[0]
					w_list = [W]
				else: # Loop files
					t_ids = np.loadtxt(f"{z}.ids", dtype=np.str_)
					assert np.sum(z_ids != t_ids) == 0, \
						"Samples do not match across files!"
					k_tmp = np.loadtxt(f"{z}.win", dtype=np.uint8, usecols=[5])
					k_vec = np.append(k_vec, k_tmp)
					W += k_tmp.shape[0]
					w_list.append(k_tmp.shape[0])
		w_vec = np.array(w_list, dtype=int)
		del t_ids, k_tmp, w_list
	else: # Single file (chromosome)
		Z_list = [args.clusters]
		assert os.path.isfile(f"{Z_list[0]}.bca"), "bca file doesn't exist!"
		assert os.path.isfile(f"{Z_list[0]}.ids"), "ids file doesn't exist!"
		assert os.path.isfile(f"{Z_list[0]}.win"), "win file doesn't exist!"
		z_ids = np.loadtxt(f"{Z_list[0]}.ids", dtype=np.str_)
		k_vec = np.loadtxt(f"{Z_list[0]}.win", dtype=np.uint8, usecols=[5])
		n = z_ids.shape[0]
		W = k_vec.shape[0]
		w_vec = np.array([W], dtype=int)
	print(f"Parsing {len(Z_list)} file(s).")
	
	# Estimate genome-wide relationship matrix
	if args.grm:
		print("Estimating genome-wide relationship matrix (GRM).")
		s_pre = ""
		K = 0
		M = 0.0
		G = np.zeros((n, n), dtype=np.float32)
		for z in np.arange(len(Z_list)): # Loop through files
			# Print information
			s_beg = f"Processing file {z+1}/{len(Z_list)}"
			print(f"\r{s_beg: <{len(s_pre)}}", end="")

			# Load haplotype cluster assignment file
			with open(f"{Z_list[z]}.bca", "rb") as f:
				# Check magic numbers
				m_vec = np.fromfile(f, dtype=np.uint8, count=3)
				assert np.allclose(m_vec, np.array([7, 9, 13], dtype=np.uint8)), \
					"Magic number doesn't match file format!"
				
				# Add haplotype cluster assignments to container
				Z_tmp = np.fromfile(f, dtype=np.uint8)
				Z_tmp.shape = (w_vec[z], 2*n)
			del m_vec

			# File setup
			k_tmp = k_vec[K:(K + w_vec[z])]
			m = np.sum(k_tmp, dtype=int)

			# Aggregate alleles and estimate cluster frequencies
			Z_agg = np.zeros((m, n), dtype=np.uint8)
			p_tmp = np.zeros(m, dtype=np.float32)
			shared_cy.haplotypeAggregate(Z_tmp, Z_agg, p_tmp, k_tmp)
			del Z_tmp, k_tmp

			# Setup GRM part settings
			B = ceil(m/args.batch) # Number of batches
			a_tmp = np.ones(m, dtype=np.float32)
			Z_bat = np.zeros((args.batch, n), dtype=np.float32)

			# Estimate GRM part in batches
			for b in np.arange(B):
				s_bat = f"{s_beg}. Batch {b+1}/{B}"
				print(f"\r{s_bat}", end="") # Print information
				m_b = b*args.batch
				if b == (B-1): # Last batch
					Z_bat = np.zeros((m - m_b, n), dtype=np.float32)
				shared_cy.batchZ(Z_agg, Z_bat, p_tmp, a_tmp, m_b, args.threads)

				# Aggregate across batches
				G += np.dot(Z_bat.T, Z_bat)
			K += w_vec[z]
			M += np.sum(p_tmp*(1.0 - p_tmp), dtype=float)
			s_pre = s_bat
			del a_tmp, p_tmp, Z_agg, Z_bat
		G *= (1.0/(2.0*M))
		print(".\n")
		
		# Gower and data centering of GRM
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
		z_ids = z_ids.reshape(-1,1)
		if args.duplicate_fid:
			fam = z_ids.repeat(2, axis=1)
		else:
			fam = np.hstack((np.zeros((n, 1), dtype=np.uint8), z_ids))
		np.savetxt(f"{args.out}.grm.id", fam, delimiter="\t", fmt="%s")
		print("Saved genome-wide relationship matrix in GCTA format:\n" + \
			f"- {args.out}.grm.bin\n" + \
			f"- {args.out}.grm.N.bin\n" + \
			f"- {args.out}.grm.id\n")
		del G, fam
	
	# Infer population structure using PCA
	if args.pca is not None:
		# Load haplotype cluster assignments from binary hapla format
		B = 0
		Z = np.zeros((W, 2*n), dtype=np.uint8)
		for z in np.arange(len(Z_list)):
			with open(f"{Z_list[z]}.bca", "rb") as f:
				# Check magic numbers
				m_vec = np.fromfile(f, dtype=np.uint8, count=3)
				assert np.allclose(m_vec, np.array([7, 9, 13], dtype=np.uint8)), \
					"Magic number doesn't match file format!"
				
				# Add haplotype cluster assignments to container
				z_tmp = np.fromfile(f, dtype=np.uint8)
				z_tmp.shape = (w_vec[z], 2*n)
				Z[B:(B + w_vec[z]),:] = z_tmp
				B += w_vec[z]
			print(f"\rParsed file {z+1}/{len(Z_list)}", end="")
		del m_vec, z_tmp

		# Count haplotype cluster alleles
		m = np.sum(k_vec, dtype=int)

		# Print information
		print(f"\rLoaded haplotype cluster assignments:\n" + \
			f"- {n} samples\n" + \
			f"- {W} windows\n" + \
			f"- {m} clusters\n")
		
		# Populate full matrix and estimate cluster frequencies
		Z_agg = np.zeros((m, n), dtype=np.uint8)
		p_vec = np.zeros(m, dtype=np.float32)
		shared_cy.haplotypeAggregate(Z, Z_agg, p_vec, k_vec)
		del Z, k_vec
		a_vec = np.sqrt(2.0*p_vec*(1-p_vec))

		# Randomized SVD
		if args.randomized:
			print(f"Computing randomized SVD, extracting {args.pca} eigenvectors.")

			# Randomized SVD in batches
			U, S, V = functions.randomizedSVD(Z_agg, p_vec, a_vec, args.pca, \
				args.batch, args.threads)

			# Save matrices
			if args.raw: # Only eigenvectors
				np.savetxt(f"{args.out}.eigenvecs", V, fmt="%.7f")
			else: # Include FID and IID fields
				z_ids = z_ids.reshape(-1,1)
				if args.duplicate_fid:
					fam = z_ids.repeat(2, axis=1)
				else:
					fam = np.hstack((np.zeros((n, 1), dtype=np.uint8), z_ids))
				V = np.hstack((fam, np.round(V, 7)))
				h = ["#FID", "IID"] + [f"PC{k}" for k in range(1, args.pca+1)]
				np.savetxt(f"{args.out}.eigenvecs", V, fmt="%s", delimiter="\t", \
					header="\t".join(h), comments="")
			np.savetxt(f"{args.out}.eigenvals", (S*S)/float(m), fmt="%.7f")
			if args.loadings:
				np.savetxt(f"{args.out}.loadings", U, fmt="%.7f")
				print(f"Saved loadings as {args.out}.loadings")
			del z_ids, Z_agg, p_vec, a_vec, U, S, V, h, fam
		else: # Truncated SVD
			print(f"Computing truncated SVD, extracting {args.pca} eigenvectors.")

			# Standardize
			Z_std = np.zeros((m, n), dtype=np.float32)
			shared_cy.standardizeZ(Z_agg, Z_std, p_vec, a_vec, args.threads)
			del Z_agg

			# Truncated SVD (Arnoldi)
			U, S, Vt = svds(Z_std, k=args.pca)
			S = S[::-1]
			V = Vt[::-1,:].T
			del Z_std, Vt

			# Save matrices
			if args.raw: # Only eigenvectors
				np.savetxt(f"{args.out}.eigenvecs", V, fmt="%.7f")
			else: # Include FID and IID fields
				z_ids = z_ids.reshape(-1,1)
				if args.duplicate_fid:
					fam = z_ids.repeat(2, axis=1)
				else:
					fam = np.hstack((np.zeros((n, 1), dtype=np.uint8), z_ids))
				V = np.hstack((fam, np.round(V, 7)))
				h = ["#FID", "IID"] + [f"PC{k}" for k in range(1, args.pca+1)]
				np.savetxt(f"{args.out}.eigenvecs", V, fmt="%s", delimiter="\t", \
					header="\t".join(h), comments="")
			np.savetxt(f"{args.out}.eigenvals", (S*S)/float(m), fmt="%.7f")
			if args.loadings:
				np.savetxt(f"{args.out}.loadings", U, fmt="%.7f")
				print(f"Saved loadings as {args.out}.loadings")
			del z_ids, p_vec, a_vec, U, S, V, h, fam
		print(f"Saved eigenvectors as {args.out}.eigenvecs")
		print(f"Saved eigenvalues as {args.out}.eigenvals\n")

	# Print elapsed time for computation
	t_tot = time()-start
	t_min = int(t_tot//60)
	t_sec = int(t_tot - t_min*60)
	print(f"Total elapsed time: {t_min}m{t_sec}s")



##### Main exception #####
assert __name__ != "__main__", "Please use the 'hapla struct' command!"
