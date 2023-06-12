"""
hapla.
Estimate polygenic risk scores using summary statistics.
"""

__author__ = "Jonas Meisner"

# Libraries
import os

##### hapla prs #####
def main(args):
	print("hapla prs by Jonas Meisner (v0.1)")
	print(f"Using {args.threads} thread(s).")

	# Check input
	assert (args.filelist is not None) or (args.clusters is not None), \
		"No input data (--filelist or --clusters)!"
	assert args.assoc is not None, \
		"Please provide summary statistics (--assoc)!"

	# Control threads of external numerical libraries
	os.environ["MKL_NUM_THREADS"] = str(args.threads)
	os.environ["OMP_NUM_THREADS"] = str(args.threads)
	os.environ["NUMEXPR_NUM_THREADS"] = str(args.threads)
	os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)

	# Import numerical libraries and cython functions
	import numpy as np
	from src import assoc_cy

	### Load data
	# Load haplotype cluster assignments (and concatentate across windows)
	if args.filelist is not None:
		Z_list = []
		with open(args.filelist) as f:
			N_chr = 1
			for chr in f:
				Z_list.append(np.load(chr.strip("\n")))
				print(f"\rParsed file #{N_chr}", end="")
				N_chr += 1
		Z_mat = np.concatenate(Z_list, axis=0)
		del Z_list
	else:
		Z_mat = np.load(args.clusters)
	W = Z_mat.shape[0]
	n = Z_mat.shape[1]//2
	print("\rLoaded haplotype cluster assignments of " + \
		f"{int(2*n)} haplotypes in {W} windows.")

	# Load summary statistics
	w, b = np.loadtxt(args.assoc, dtype=float, unpack=True, usecols=[1,4], \
		skiprows=1)
	w = w.astype(int)
	
	# Extract number of clusters in each window
	K_vec = np.zeros(W, dtype=np.uint8)
	W_assoc = assoc_cy.updateK(K_vec, w)
	assert W == W_assoc, "Number of windows differ between files!"

	# Populate haplotype cluster matrix
	Z_tilde = np.zeros((np.sum(K_vec), n), dtype=np.uint8)
	assoc_cy.updateZ(Z_mat, Z_tilde, K_vec)
	del Z_mat

	### Estimate polygenic risk score
	y_prs = np.dot(Z_tilde.T, b)/float(2.0*W)
	np.savetxt(f"{args.out}.prs", y_prs, fmt="%.7f")
	print(f"Saved PRS from summary statistics as {args.out}.prs")



##### Main exception #####
assert __name__ != "__main__", "Please use the 'hapla prs' command!"
