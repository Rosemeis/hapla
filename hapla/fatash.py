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
	print("hapla by Jonas Meisner (v0.9)")
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
	import scipy.optimize as optim
	from hapla import fatash_cy
	from hapla import functions

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

	# Containers
	v = np.zeros(K) # Help vector
	T = np.zeros((K, K)) # Transitions

	# Loop over chromosomes
	W_tot = 0
	print(f"Inferring local ancestry tracts with {K} ancestral sources.\n")
	for c in np.arange(n_chr):
		print(f"Chromsome {c+1}/{n_chr}")
		s_chr = time()

		# Load haplotype assignments and log P matrix
		Z = np.ascontiguousarray(np.load(Z_list[c]).T)
		W = Z.shape[1]
		P_chr = np.ascontiguousarray(np.swapaxes(P[W_tot:(W_tot + W)], 1, 2))

		# Setup parameters and alpha optimization
		if c == 0:
			n = Z.shape[0]
			if Q.shape[0] == n//2:
				N = 2
			else:
				N = 1
				assert Q.shape[0] == n, "Number of samples do not match!"
			if args.optim: # Individual alpha rates
				a = np.zeros(n)
		else:
			assert Z.shape[0] == n, "Number of samples do not match!"
		assert P_chr.shape[1] >= (np.max(Z) + 1), "Number of clusters do not match!"

		# Containers
		E = np.zeros((n, W, K)) # Emission probabilities
		L = np.zeros((n, W, K)) # Posterior probabilities
		A = np.zeros((W, K)) # Forward matrix
		B = np.zeros((W, K)) # Backward matrix

		# Compute emission probabilities
		fatash_cy.calcEmissions(Z, P_chr, E, args.threads)
		del Z, P_chr

		# HMM for each haplotype
		for i in range(n):
			print(f"\rHaplotype {i+1}/{n}", end="")

			# Optimize alpha parameter
			if args.optim:
				opt = optim.minimize_scalar(
					fun=functions.loglikeWrapper,
					args=(E, Q, T, A, v, N, i),
					method="bounded",
					bounds=tuple(args.alpha_bound)
				)
				alpha = opt.x
				a[i] = alpha
			else:
				alpha = args.alpha

			# Compute probabilities
			fatash_cy.calcTransition(T, Q, i//N, alpha)
			fatash_cy.calcFwdBwd(E, L, Q, T, A, B, v, N, i)
		print(".")

		# Save matrices
		if n_chr == 1:
			np.savetxt(f"{args.out}.path", L.argmax(axis=2), fmt="%i")
			print(f"Saved posterior decoding path as {args.out}.path")
			if args.optim:
				np.savetxt(f"{args.out}.alpha", a, fmt="%.6f")
				print(f"Saved individual alpha rates as {args.out}.alpha")
			print("")
		else:
			np.savetxt(f"{args.out}.chr{c+1}.path", L.argmax(axis=2), fmt="%i")
			print(f"Saved posterior decoding path as {args.out}.chr{c+1}.path")
			if args.optim:
				np.savetxt(f"{args.out}.chr{c+1}.alpha", a, fmt="%.6f")
				print(f"Saved individual alpha rates as {args.out}.chr{c+1}.alpha")
			
			# Print elapsed time of chromosome 
			t_chr = time()-s_chr
			t_min = int(t_chr//60)
			t_sec = int(t_chr - t_min*60)
			print(f"Elapsed time: {t_min}m{t_sec}s\n")
		W_tot += W
		del E, L, A, B
	assert P.shape[0] == W_tot, "Number of windows did not match!"

	# Print elapsed time for computation
	t_tot = time()-start
	t_min = int(t_tot//60)
	t_sec = int(t_tot - t_min*60)
	print(f"Total elapsed time: {t_min}m{t_sec}s")



##### Main exception #####
assert __name__ != "__main__", "Please use the 'hapla fatash' command!"
