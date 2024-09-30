"""
hapla.
Infer local ancestry tracts.
"""

__author__ = "Jonas Meisner"

# Libraries
import os
from time import time

##### hapla fatash #####
def main(args):
	print("-----------------------------------")
	print("hapla by Jonas Meisner (v0.11)")
	print(f"hapla fatash using {args.threads} thread(s)")
	print("-----------------------------------\n")

	# Check input
	assert (args.filelist is not None) or (args.clusters is not None), \
		"No input data (--filelist or --clusters)!"
	assert args.pfile is not None, "No P-matrix provided (--p-matrix)!"
	assert args.qfile is not None, "No Q-matrix provided (--q-matrix)!"
	start = time()

	# Control threads of external numerical libraries
	os.environ["MKL_NUM_THREADS"] = str(args.threads)
	os.environ["OMP_NUM_THREADS"] = str(args.threads)
	os.environ["NUMEXPR_NUM_THREADS"] = str(args.threads)
	os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)

	# Import numerical libraries and cython functions
	import numpy as np
	from hapla import fatash_cy

	# Load data (and concatentate across windows)
	if args.filelist is not None:
		Z_list = []
		with open(args.filelist) as f:
			for z_file in f:
				Z_list.append(z_file.strip("\n"))
	else:
		Z_list = [args.clusters]
	n_chr = len(Z_list)
	print(f"Parsing {n_chr} file(s).")

	# Load P and Q matrices
	P = np.load(args.pfile)
	Q = np.loadtxt(args.qfile, dtype=float)
	assert P.shape[1] == Q.shape[1], "Number of ancestral sources do not match!"
	K = P.shape[1]

	# General containers
	T = np.zeros((K, K)) # Transitions
	v = np.zeros(K) # Help vector

	# Loop over chromosomes
	W_tot = 0
	print(f"Inferring local ancestry tracts with {K} ancestral sources.\n")
	for chrom in np.arange(n_chr):
		print(f"Chromsome {chrom+1}/{n_chr}")
		s_chr = time()

		# Load haplotype assignments and log P matrix
		Z_chr = np.ascontiguousarray(np.load(Z_list[chrom]).T)
		W_chr = Z_chr.shape[1]
		P_chr = np.ascontiguousarray(np.swapaxes(P[W_tot:(W_tot + W_chr)], 1, 2))

		# Setup parameters
		if chrom == 0:
			n = Z_chr.shape[0]
			if Q.shape[0] == n//2:
				N = 2
			else:
				N = 1
				assert Q.shape[0] == n, "Number of samples do not match!"
		else:
			assert Z_chr.shape[0] == n, "Number of samples do not match!"
		assert P_chr.shape[1] > np.max(Z_chr), "Number of clusters do not match!"

		# Containers
		E = np.zeros((n, W_chr, K)) # Emission probabilities
		A = np.zeros((W_chr, K)) # Forward matrix
		if args.viterbi:
			I = np.zeros((W_chr, K), dtype=np.uint8) # Index matrix
			V = np.zeros((n, W_chr), dtype=np.uint8) # Viterbi path
		else:
			c = np.zeros(W_chr) # Help vector
			B = np.zeros((W_chr, K)) # Backward matrix
			L = np.zeros((n, W_chr, K)) # Posterior probabilities

		# Compute emission probabilities
		fatash_cy.calcEmissions(Z_chr, P_chr, E, args.threads)
		del Z_chr, P_chr

		# HMM for each haplotype
		for i in np.arange(n):
			print(f"\rHaplotype {i+1}/{n}", end="")
			fatash_cy.calcTransition(T, Q, i//N, args.alpha)

			# Compute Viterbi and decoding
			if args.viterbi:
				fatash_cy.viterbi(E, Q, T, A, I, V, N, i)
			else: # Compute posterior probabilities
				fatash_cy.calcFwdBwd(E, L, Q, T, A, B, c, v, N, i)
		print(".")

		# Save matrices
		if n_chr == 1:
			if args.viterbi:
				np.savetxt(f"{args.out}.path", V, fmt="%i")
				print(f"Saved Viterbi decoding path as {args.out}.path")
			else:
				np.savetxt(f"{args.out}.path", L.argmax(axis=2), fmt="%i")
				print(f"Saved posterior decoding path as {args.out}.path")
				if args.save_posterior:
					np.save(f"{args.out}.post", L)
					print(f"Saved posteriors as {args.out}.post.npy")
			print("")
		else:
			if args.viterbi:
				np.savetxt(f"{args.out}.chr{chrom+1}.path", V, fmt="%i")
				print(f"Saved Viterbi decoding as {args.out}.chr{chrom+1}.path")
			else:
				np.savetxt(f"{args.out}.chr{chrom+1}.path", L.argmax(axis=2), fmt="%i")
				print(f"Saved posterior decoding as {args.out}.chr{chrom+1}.path")
				if args.save_posterior:
					np.save(f"{args.out}.chr{chrom+1}.post", L)
					print(f"Saved posteriors as {args.out}.chr{chrom+1}.post.npy")

			# Print elapsed time of chromosome 
			t_chr = time()-s_chr
			t_min = int(t_chr//60)
			t_sec = int(t_chr - t_min*60)
			print(f"Elapsed time: {t_min}m{t_sec}s\n")
		W_tot += W_chr
		del E, A
		if args.viterbi:
			del I, V
		else:
			del c, B, L
	assert P.shape[0] == W_tot, "Number of windows did not match!"

	# Print elapsed time for computation
	t_tot = time()-start
	t_min = int(t_tot//60)
	t_sec = int(t_tot - t_min*60)
	print(f"Total elapsed time: {t_min}m{t_sec}s")



##### Main exception #####
assert __name__ != "__main__", "Please use the 'hapla fatash' command!"
