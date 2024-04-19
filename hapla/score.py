"""
hapla.
Estimate cluster effects and genomic BLUPs using haplotype cluster alleles.
"""

__author__ = "Jonas Meisner"

# Libraries
import os
from time import time

##### hapla score #####
def main(args):
	print("-----------------------------------")
	print("hapla by Jonas Meisner (v0.7)")
	print(f"hapla score using {args.threads} thread(s)")
	print("-----------------------------------\n")

	# Check input
	assert (args.filelist is not None) or (args.clusters is not None), \
		"No input data (--filelist or --clusters)!"
	if args.effects is not None:
		assert args.predict, "Please toggle prediction (--predict)!"
	if (args.effects is None) & (args.predict):
		print("Estimating cluster effects and predicting in same data!\n")
	start = time()

	# Control threads of external numerical libraries
	os.environ["MKL_NUM_THREADS"] = str(args.threads)
	os.environ["OMP_NUM_THREADS"] = str(args.threads)
	os.environ["NUMEXPR_NUM_THREADS"] = str(args.threads)
	os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)

	# Import numerical libraries and cython functions
	import numpy as np
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

	# Load cvBLUPs
	y = np.loadtxt(args.blups, dtype=np.float32)
	n = y.shape[0]

	# Estimate cluster effects if none provided
	if args.effects is None:
		print("Estimating cluster effects.")
		print("Computing symmetric matrix.")
		M = 0
		G = np.zeros((n, n), dtype=np.float32)
		s_pre = ""
		for z in np.arange(len(Z_list)): # Loop through files
			# Print information
			s_beg = f"Processing file {z+1}/{len(Z_list)}"
			print(f"\r{s_beg: <{len(s_pre)}}", end="")

			# Load file
			Z_mat = np.load(Z_list[z])
			assert n == Z_mat.shape[1]//2, \
				"Number of samples differ between cluster files!"
			W = Z_mat.shape[0]

			# Count haplotype cluster alleles and find rarest clusters
			R_vec = np.zeros(W, dtype=np.uint8)
			K_vec = np.max(Z_mat, axis=1) + 1
			shared_cy.findRare(Z_mat, R_vec, K_vec, args.threads)
			m = np.sum(K_vec-1, dtype=int)

			# Populate full matrix and estimate cluster frequencies
			Z = np.zeros((m, n), dtype=np.uint8)
			p = np.zeros(m, dtype=np.float32)
			shared_cy.haplotypeAggregate(Z_mat, Z, p, R_vec, K_vec)
			del Z_mat

			# Setup GRM part settings
			B = m//args.batch # Number of batches
			a = np.power(2*p*(1-p), 0.5*args.alpha)
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
			del a, p, Z, Z_b, R_vec, K_vec
		print(".\n")

		# (A^T A)^{-1}
		print("Computing inverse matrix.")
		G = np.linalg.pinv(G, hermitian=True)
		h = np.dot(G, y)
		del G

		# A (A^T A)^{-1} g
		print("Computing effects.")
		M_z = 0
		beta = np.zeros((M, 2), dtype=np.float32)
		s_pre = ""
		for z in np.arange(len(Z_list)): # Loop through files
			# Print information
			s_beg = f"Processing file {z+1}/{len(Z_list)}"
			print(f"\r{s_beg: <{len(s_pre)}}", end="")

			# Load file
			Z_mat = np.load(Z_list[z])
			W = Z_mat.shape[0]

			# Count haplotype cluster alleles and find rarest clusters
			R_vec = np.zeros(W, dtype=np.uint8)
			K_vec = np.max(Z_mat, axis=1) + 1
			shared_cy.findRare(Z_mat, R_vec, K_vec, args.threads)
			m = np.sum(K_vec-1, dtype=int)

			# Populate full matrix and estimate cluster frequencies
			Z = np.zeros((m, n), dtype=np.uint8)
			p = np.zeros(m, dtype=np.float32)
			shared_cy.haplotypeAggregate(Z_mat, Z, p, R_vec, K_vec)
			beta[M_z:(M_z + m):,0] = np.copy(p)
			del Z_mat

			# Setup matrix-vector multiplication part settings
			B = m//args.batch # Number of batches
			Z_b = np.zeros((args.batch, n), dtype=np.float32)

			# Estimate GRM part in batches
			for b in np.arange(B):
				s_bat = f"{s_beg}. Batch {b+1}/{B}"
				print(f"\r{s_bat}", end="") # Print information
				m_b = b*args.batch
				if b == (B-1): # Last batch
					Z_b = np.zeros((m - m_b, n), dtype=np.float32)
				shared_cy.batchZ(Z, Z_b, p, a, m_b, args.threads)

				# Compute allele effects
				beta[M_z:(M_z + Z_b.shape[0]):,1] = np.dot(Z_b, h)
				M_z += Z_b.shape[0]
			s_pre = s_bat
			del a, p, Z, Z_b, R_vec, K_vec
		print(".\n")
		
		# Save cluster effects
		np.savetxt(f"{args.out}.effects", beta, fmt="%.7f")
		print(f"Saved eigenvalues as {args.out}.effects\n")
	else:
		# Load cluster effects
		beta = np.loadtxt(args.effects, dtype=np.float32)
			
	# Estimate polygenic scores
	if args.predict:
		print("Computing polygenic scores.")
		M_z = 0
		p = beta[:,0]
		a = np.power(2*p*(1-p), 0.5*args.alpha)
		g = np.zeros(n, dtype=np.float32)
		s_pre = ""
		for z in np.arange(len(Z_list)): # Loop through files
			# Print information
			s_beg = f"Processing file {z+1}/{len(Z_list)}"
			print(f"\r{s_beg: <{len(s_pre)}}", end="")

			# Load file
			Z_mat = np.load(Z_list[z])
			assert n == Z_mat.shape[1]//2, \
				"Number of samples differ between cluster files!"
			W = Z_mat.shape[0]

			# Count haplotype cluster alleles and find rarest clusters
			R_vec = np.zeros(W, dtype=np.uint8)
			K_vec = np.max(Z_mat, axis=1) + 1
			shared_cy.findRare(Z_mat, R_vec, K_vec, args.threads)
			m = np.sum(K_vec-1, dtype=int)

			# Populate full matrix and estimate cluster frequencies
			Z = np.zeros((m, n), dtype=np.uint8)
			shared_cy.scoreAggregate(Z_mat, Z, R_vec, K_vec)
			del Z_mat

			# Setup GRM part settings
			B = m//args.batch # Number of batches
			Z_b = np.zeros((args.batch, n), dtype=np.float32)

			# Estimate GRM part in batches
			for b in np.arange(B):
				s_bat = f"{s_beg}. Batch {b+1}/{B}"
				print(f"\r{s_bat}", end="") # Print information
				m_b = b*args.batch
				if b == (B-1): # Last batch
					Z_b = np.zeros((m - m_b, n), dtype=np.float32)
				shared_cy.scoreZ(Z, Z_b, p, a, m_b, M_z, args.threads)

				# Aggregate across batches
				g += np.dot(Z_b.T, beta[M_z:(M_z + Z_b.shape[0]):,1])
				M_z += Z_b.shape[0]
			del a, Z, Z_b, R_vec, K_vec
		print(".\n")

		# Save polygenic scores
		np.savetxt(f"{args.out}.scores", g, fmt="%.7f")
		print(f"Saved polygenic scores as {args.out}.scores\n")

	# Print elapsed time for computation
	t_tot = time()-start
	t_min = int(t_tot//60)
	t_sec = int(t_tot - t_min*60)
	print(f"Total elapsed time: {t_min}m{t_sec}s")



##### Main exception #####
assert __name__ != "__main__", "Please use the 'hapla score' command!"
