"""
hapla.
Population structure inference using haplotype cluster alleles.
"""

__author__ = "Jonas Meisner"

# Libraries
import os
from datetime import datetime
from time import time

VERSION = "0.32.1"

##### hapla struct #####
def main(args, deaf):
	print("-----------------------------------")
	print(f"hapla by Jonas Meisner (v{VERSION})")
	print(f"hapla struct using {args.threads} thread(s)")
	print("-----------------------------------\n")

	# Check input
	assert (args.filelist is not None) or (args.clusters is not None), "No input data (--filelist or --clusters)!"
	assert (args.grm or (args.pca is not None)), "No analysis selected (--grm, --pca)!"
	if args.pca is not None:
		assert args.pca > 0, "Please select a valid number of eigenvectors!"
	assert args.threads > 0, "Please select a valid number of threads!"
	assert args.chunk > 0, "Please select a valid chunk size!"
	assert args.power > 0, "Please select a valid number of power iterations!"
	assert args.seed >= 0, "Please select a valid seed!"
	start = time()

	# Create log-file of used arguments
	full = vars(args)
	with open(f"{args.out}.log", "w") as log:
		log.write(f"hapla v{VERSION}\n")
		log.write("hapla struct\n")
		log.write(f"Time: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
		log.write(f"Directory: {os.getcwd()}\n")
		log.write("Options:\n")
		for key in full:
			if full[key] != deaf[key]:
				if type(full[key]) is bool:
					log.write(f"\t--{key}\n")
				else:
					log.write(f"\t--{key} {full[key]}\n")
	del full, deaf

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
					z_ids = np.genfromtxt(f"{z}.ids", dtype=np.str_)
					k_vec = np.genfromtxt(f"{z}.win", dtype=np.uint32, usecols=[5])
					N = z_ids.shape[0]
					W = k_vec.shape[0]
					w_list = [W]
				else: # Loop files
					t_ids = np.genfromtxt(f"{z}.ids", dtype=np.str_)
					assert np.sum(z_ids != t_ids) == 0, "Samples don't match across files!"
					k_tmp = np.genfromtxt(f"{z}.win", dtype=np.uint32, usecols=[5])
					k_vec = np.append(k_vec, k_tmp)
					W += k_tmp.shape[0]
					w_list.append(k_tmp.shape[0])
		F = len(Z_list)
		w_vec = np.array(w_list, dtype=np.uint32)
		del t_ids, k_tmp, w_list
	else: # Single file (chromosome)
		F = 1
		Z_list = [args.clusters]
		assert os.path.isfile(f"{Z_list[0]}.bca"), "bca file doesn't exist!"
		assert os.path.isfile(f"{Z_list[0]}.ids"), "ids file doesn't exist!"
		assert os.path.isfile(f"{Z_list[0]}.win"), "win file doesn't exist!"
		z_ids = np.genfromtxt(f"{Z_list[0]}.ids", dtype=np.str_)
		k_vec = np.genfromtxt(f"{Z_list[0]}.win", dtype=np.uint32, usecols=[5])
		N = z_ids.shape[0]
		W = k_vec.shape[0]
		w_vec = np.array([W], dtype=np.uint32)
	print(f"Parsing {F} file(s).")
	
	# Estimate genome-wide relationship matrix
	if args.grm:
		print("Estimating genome-wide relationship matrix (GRM).")
		s_pre = ""
		K = 0
		D = 0.0
		G = np.zeros((N, N), dtype=np.float32)
		for z in np.arange(F): # Loop through files
			# Print information
			s_beg = f"Processing file {z + 1}/{F}"
			print(f"\r{s_beg: <{len(s_pre)}}", end="")

			# Load haplotype cluster assignment file
			with open(f"{Z_list[z]}.bca", "rb") as f:
				# Check magic numbers
				magic = np.fromfile(f, dtype=np.uint8, count=3)
				assert np.allclose(magic, np.array([7, 9, 13], dtype=np.uint8)), \
					"Magic number doesn't match file format!"
				
				# Add haplotype cluster assignments to container
				Z_tmp = np.fromfile(f, dtype=np.uint8)
				Z_tmp.shape = (w_vec[z], 2*N)
			del magic

			# File setup
			k_tmp = k_vec[K:(K + w_vec[z])]
			c_tmp = np.insert(np.cumsum(k_tmp, dtype=np.uint32), 0, 0)
			M = np.sum(k_tmp, dtype=np.uint32)

			# Aggregate alleles and estimate cluster frequencies
			Z_agg = np.zeros((M, N), dtype=np.uint8)
			p_tmp = np.zeros(M, dtype=np.float32)
			shared_cy.haplotypeAggregate(Z_tmp, Z_agg, p_tmp, k_tmp, c_tmp)
			del Z_tmp, k_tmp, c_tmp

			# Setup GRM part settings
			B = ceil(M/args.chunk) # Number of chunks
			X = np.zeros((args.chunk, N), dtype=np.float32)

			# Estimate GRM part in chunks
			for b in np.arange(B):
				s_bat = f"{s_beg}. Chunk {b + 1}/{B}"
				print(f"\r{s_bat}", end="") # Print information
				M_b = b*args.chunk
				if b == (B - 1): # Last chunk
					X = np.zeros((M - M_b, N), dtype=np.float32)
				shared_cy.centerZ(Z_agg, X, p_tmp, M_b)

				# Aggregate across chunks
				G += np.dot(X.T, X)
			K += w_vec[z]
			D += np.sum(p_tmp*(1.0 - p_tmp), dtype=float)
			s_pre = s_bat
			del p_tmp, Z_agg, X
		G *= (1.0/(2.0*D))
		print(".\n")
		
		# Gower and data centering of GRM
		if not args.no_centering:
			print("Centering GRM.")
			u = np.mean(G, axis=1)
			G -= u.reshape(1, N)
			u = np.mean(G, axis=1)
			G -= u.reshape(N, 1)

			# Gower centering
			G *= float(N - 1)/np.trace(G)
			del u

		# Save matrix
		G = G[np.tril_indices(N)]
		G.tofile(f"{args.out}.grm.bin")
		np.full(N*(N + 1)//2, D, dtype=np.float32).tofile(f"{args.out}.grm.N.bin")
		z_ids = z_ids.reshape(-1, 1)
		if args.duplicate_fid:
			fam = z_ids.repeat(2, axis=1)
		else:
			fam = np.hstack((np.zeros((N, 1), dtype=np.uint8), z_ids))
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
		Z = np.zeros((W, 2*N), dtype=np.uint8)
		for z in np.arange(F):
			with open(f"{Z_list[z]}.bca", "rb") as f:
				# Check magic numbers
				magic = np.fromfile(f, dtype=np.uint8, count=3)
				assert np.allclose(magic, np.array([7, 9, 13], dtype=np.uint8)), \
					"Magic number doesn't match file format!"
				
				# Add haplotype cluster assignments to container
				z_tmp = np.fromfile(f, dtype=np.uint8)
				z_tmp.shape = (w_vec[z], 2*N)
				Z[B:(B + w_vec[z]),:] = z_tmp
				B += w_vec[z]
			print(f"\rParsed file {z + 1}/{F}", end="")
		del magic, z_tmp

		# Count haplotype cluster alleles
		M = np.sum(k_vec, dtype=np.uint32)

		# Print information
		print(f"\rLoaded haplotype cluster assignments:\n" + \
			f"- {N} samples\n" + \
			f"- {W} windows\n" + \
			f"- {M} clusters\n")
		
		# Estimate cluster frequencies
		p_vec = np.zeros(M, dtype=np.float32)
		c_vec = np.insert(np.cumsum(k_vec, dtype=np.uint32), 0, 0)

		# Randomized SVD
		print(f"Computing randomized SVD, extracting {args.pca} eigenvectors.")
		rng = np.random.default_rng(args.seed)
		if args.memory:
			# SVD with condensed data format
			shared_cy.estimateFreq(Z, p_vec, k_vec, c_vec)
			a_vec = 1.0/np.sqrt(2.0*p_vec*(1.0 - p_vec))
			U, S, V = functions.memorySVD(Z, p_vec, a_vec, k_vec, c_vec, args.pca, args.chunk, args.power, rng)
			del Z
		else:
			# SVD with expanded data format 
			Z_agg = np.zeros((M, N), dtype=np.uint8)
			shared_cy.haplotypeAggregate(Z, Z_agg, p_vec, k_vec, c_vec)
			a_vec = 1.0/np.sqrt(2.0*p_vec*(1.0 - p_vec))
			del Z, k_vec, c_vec
			U, S, V = functions.randomizedSVD(Z_agg, p_vec, a_vec, args.pca, args.chunk, args.power, rng)
			del Z_agg
		del p_vec, a_vec
		print(".\n")

		# Save matrices
		if args.raw: # Only eigenvectors
			np.savetxt(f"{args.out}.eigenvecs", V, fmt="%.6f")
		else: # Include FID and IID fields
			z_ids = z_ids.reshape(-1, 1)
			if args.duplicate_fid:
				fam = z_ids.repeat(2, axis=1)
			else:
				fam = np.hstack((np.zeros((N, 1), dtype=np.uint8), z_ids))
			V = np.hstack((fam, np.round(V, 7)))
			h = ["#FID", "IID"] + [f"PC{k}" for k in range(1, args.pca + 1)]
			np.savetxt(f"{args.out}.eigenvecs", V, fmt="%s", delimiter="\t", comments="", header="\t".join(h))
		print(f"Saved eigenvectors as {args.out}.eigenvecs")
		np.savetxt(f"{args.out}.eigenvals", (S*S)/float(M), fmt="%.6f")
		print(f"Saved eigenvalues as {args.out}.eigenvals")
		if args.loadings:
			np.savetxt(f"{args.out}.loadings", U, fmt="%.6f")
			print(f"Saved loadings as {args.out}.loadings")
		print("")

	# Print elapsed time for computation
	t_tot = time() - start
	t_min = int(t_tot//60)
	t_sec = int(t_tot - t_min*60)
	print(f"Total elapsed time: {t_min}m{t_sec}s")

	# Write to log-file
	with open(f"{args.out}.log", "a") as log:
		if args.grm:
			log.write("\nSaved genome-wide relationship matrix in GCTA format:\n" + \
				f"- {args.out}.grm.bin\n" + \
				f"- {args.out}.grm.N.bin\n" + \
				f"- {args.out}.grm.id\n")
		if args.pca is not None:
			log.write(f"\nSaved eigenvectors as {args.out}.eigenvecs\n")
			log.write(f"Saved eigenvalues as {args.out}.eigenvals\n")
			if args.loadings:
				log.write(f"Saved loadings as {args.out}.loadings\n")
		log.write(f"\nTotal elapsed time: {t_min}m{t_sec}s\n")



##### Main exception #####
assert __name__ != "__main__", "Please use the 'hapla struct' command!"
